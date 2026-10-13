"""
Step 6: re-run the classification study on the leakage-safe (grouped) split.

No notebook needed. It rebuilds the same feature sets as 03_multimodal_fusion
(78 audio / 15 handcrafted lyric / 384 embedding / fused 477), uses the same five
classifiers and grids, but:
  * train / validation / test come from outputs/revision/splits.csv (no artist or
    near-duplicate-lyric group crosses a split),
  * the inner GridSearchCV is grouped (StratifiedGroupKFold),
  * the shuffled-label control permutes the TRAINING labels only,
  * confidence intervals use a group bootstrap,
  * a "file-format only" baseline (codec / bitrate / sample rate / channels) is
    scored on the same split, so the paper can show how much of the result the
    acquisition confound alone explains.

Run from the project folder (the one containing outputs/):
    python scripts/run_grouped_classifiers.py --quick --audio original      # about 5-10 min
    python scripts/run_grouped_classifiers.py --audio original               # full grids
    python scripts/run_grouped_classifiers.py --audio common                 # harmonised audio
    python scripts/run_grouped_classifiers.py --audio matched

Needs: scikit-learn, xgboost, pandas, numpy (already installed for the notebooks).
Outputs go to outputs/revision/grouped_<audio>_*.csv and are printed at the end.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from common import (OUTPUT_DIR, REVISION_DIR, RANDOM_SEED, load_embeddings,
                          load_lyrics_ok, preflight, true_mask)

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------ features (as in the notebook)
META = {
    "song_id", "title", "artist", "language", "final_genre", "source", "audio_path", "original_audio_path",
    "ok", "error", "is_mono", "is_clipped", "transcript_ok", "transcript_error", "full_text", "clean_text",
    "transcript_path", "vocals_path", "demucs_used", "whisper_lang", "whisper_detected_lang",
    "detected_language", "duration_seconds", "duration_sec", "label", "target",
}
AUDIO_EXCLUDE = {
    "lufs_integrated", "rms_db", "sample_peak_dbfs", "true_peak_dbfs", "hard_clip_count", "clip_rate_per_sec",
    "is_clipped", "silence_ratio", "pentatonic_energy_ratio", "pitch_dev_mean_cents", "pitch_dev_std_cents",
    "pitch_microtonal_ratio", "tuning_offset_cents", "beat_regularity_cv", "mfcc_1_mean", "mfcc_1_std",
}
LYRICS_EXCLUDE = {
    "whisper_lang_match", "whisper_avg_logprob", "whisper_no_speech_ratio", "avg_logprob", "no_speech_ratio",
    "total_lines", "non_empty_lines", "segment_count", "empty_line_ratio", "avg_words_per_line",
    "max_words_per_line",
}

# paper numbers (original random split) for the comparison printout
PAPER = {"audio_only": 0.9637, "lyrics_combined": 0.9220, "fused": 0.9749}


def numeric_features(df, exclude):
    feats = []
    for col in df.columns:
        if col in META or col in exclude or col.startswith("muq_"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0 or values.isna().mean() >= 0.50:
            continue
        df[col] = values
        feats.append(col)
    return feats


def load_table(audio_tag):
    audio_path = OUTPUT_DIR / ("audio_features.csv" if audio_tag == "original"
                               else f"audio_features_{audio_tag}.csv")
    if not audio_path.exists():
        sys.exit(f"STOP: {audio_path} not found. Run harmonise_audio.py and extract_harmonised_features.py for mode '{audio_tag}' first.")
    audio = pd.read_csv(audio_path, low_memory=False)
    audio["song_id"] = audio["song_id"].astype(str).str.strip()
    if "ok" in audio.columns:
        audio = audio.loc[true_mask(audio["ok"])].copy()
    audio = audio.drop_duplicates("song_id", keep="last")
    audio = audio[audio["source"].isin(["AI", "Human"])].reset_index(drop=True)

    lyrics = load_lyrics_ok()
    emb = load_embeddings()
    a_feats = numeric_features(audio, AUDIO_EXCLUDE)
    l_feats = numeric_features(lyrics, LYRICS_EXCLUDE)
    e_feats = [c for c in emb.columns if c.startswith("emb_")]
    keep = [c for c in ["song_id", "title", "artist", "source", "language", "final_genre"] if c in audio.columns]
    merged = (audio[keep + a_feats]
              .merge(lyrics[["song_id"] + l_feats], on="song_id", how="inner")
              .merge(emb, on="song_id", how="inner")
              .drop_duplicates("song_id", keep="last").reset_index(drop=True))

    splits = pd.read_csv(REVISION_DIR / "splits.csv")
    splits["song_id"] = splits["song_id"].astype(str).str.strip()
    merged = merged.merge(splits[["song_id", "group_id", "holdout_split"] +
                                 [c for c in splits.columns if c.endswith("_fold")]], on="song_id", how="inner")
    sets = {
        "audio_only": a_feats,
        "lyrics_handcrafted": l_feats,
        "lyrics_embeddings": e_feats,
        "lyrics_combined": l_feats + e_feats,
        "fused": a_feats + l_feats + e_feats,
    }
    return merged, sets


# ------------------------------------------------------------------ models (as in the notebook)
def make_pipe(clf, scale=True):
    steps = [("imputer", SimpleImputer(strategy="median")), ("variance", VarianceThreshold(0.0))]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", clf))
    return Pipeline(steps)


def build_models(quick):
    from xgboost import XGBClassifier
    base = {
        "Logistic Regression": make_pipe(LogisticRegression(max_iter=2000, random_state=RANDOM_SEED,
                                                            class_weight="balanced")),
        "SVM (RBF)": make_pipe(SVC(probability=True, kernel="rbf", random_state=RANDOM_SEED,
                                   class_weight="balanced")),
        "Random Forest": make_pipe(RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                                          class_weight="balanced", n_jobs=1), scale=False),
        "XGBoost": make_pipe(XGBClassifier(objective="binary:logistic", n_estimators=200, learning_rate=0.1,
                                           max_depth=4, random_state=RANDOM_SEED, eval_metric="logloss",
                                           n_jobs=1), scale=False),
        "MLP": make_pipe(MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
                                       max_iter=500, early_stopping=True, validation_fraction=0.1,
                                       random_state=RANDOM_SEED)),
    }
    full = {
        "Logistic Regression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
        "SVM (RBF)": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]},
        "Random Forest": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20],
                          "clf__min_samples_split": [2, 5]},
        "XGBoost": {"clf__learning_rate": [0.05, 0.1, 0.2], "clf__max_depth": [3, 4, 6],
                    "clf__n_estimators": [100, 200]},
        "MLP": {"clf__hidden_layer_sizes": [(64, 32), (128, 64, 32), (256, 128, 64)],
                "clf__alpha": [1e-4, 1e-3, 1e-2]},
    }
    quick_grids = {
        "Logistic Regression": {"clf__C": [0.1, 1.0]},
        "SVM (RBF)": {"clf__C": [1.0, 10.0]},
        "Random Forest": {"clf__max_depth": [None, 20]},
        "XGBoost": {"clf__learning_rate": [0.1], "clf__max_depth": [4, 6]},
        "MLP": {"clf__alpha": [1e-3, 1e-2]},
    }
    return base, (quick_grids if quick else full)


def scores(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X)


def metrics(model, X, y):
    pred = model.predict(X)
    try:
        auc = roc_auc_score(y, scores(model, X))
    except Exception:
        auc = np.nan
    return {"f1_macro": f1_score(y, pred, average="macro"), "accuracy": accuracy_score(y, pred), "auroc": auc}


def group_bootstrap(y, pred, score, groups, n=1000, seed=RANDOM_SEED):
    rng = np.random.RandomState(seed)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    f1s, aucs = [], []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[idx])) < 2:
            continue
        f1s.append(f1_score(y[idx], pred[idx], average="macro"))
        aucs.append(roc_auc_score(y[idx], score[idx]))
    lo = lambda v: float(np.percentile(v, 2.5))
    hi = lambda v: float(np.percentile(v, 97.5))
    return (lo(f1s), hi(f1s)), (lo(aucs), hi(aucs))


# ------------------------------------------------------------------ file-format-only baseline
def format_baseline(merged, y, tr, te):
    path = REVISION_DIR / "B_file_metadata.csv"
    if not path.exists():
        print("  (format-only baseline skipped: run audit_dataset.py first to create B_file_metadata.csv)")
        return None
    meta = pd.read_csv(path)
    meta["song_id"] = meta["song_id"].astype(str).str.strip()
    cols = ["codec", "bitrate_kbps", "sample_rate", "channels"]
    m = merged[["song_id"]].merge(meta[["song_id"] + cols], on="song_id", how="left")
    X = m[cols].astype(str)
    model = make_pipeline(OneHotEncoder(handle_unknown="ignore"),
                          LogisticRegression(max_iter=1000, class_weight="balanced"))
    model.fit(X.iloc[tr], y[tr])
    return metrics(model, X.iloc[te], y[te])


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default="original", choices=["original", "common", "matched"])
    ap.add_argument("--quick", action="store_true", help="smaller grids (about 5x faster)")
    ap.add_argument("--sets", default="audio_only,lyrics_combined,fused",
                    help="comma list from: audio_only,lyrics_handcrafted,lyrics_embeddings,lyrics_combined,fused")
    ap.add_argument("--models", default="", help="optional comma list to run only some classifiers")
    ap.add_argument("--no-cv", action="store_true", help="skip the repeated grouped CV")
    ap.add_argument("--shuffles", type=int, default=10, help="permutations for the shuffled-label control")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--only-ids-from", default="",
                    help="CSV with a song_id column (for example outputs/audio_features_common.csv): use only those "
                         "tracks, so original and harmonised audio can be compared on exactly the same songs")
    args = ap.parse_args()

    preflight(("audio", "lyrics", "embeddings"))
    if not (REVISION_DIR / "splits.csv").exists():
        sys.exit("STOP: outputs/revision/splits.csv missing. Run make_grouped_splits.py first.")
    tag = args.audio + ("_subset" if args.only_ids_from else "") + ("_quick" if args.quick else "")
    merged, sets = load_table(args.audio)
    if args.only_ids_from:
        ids = pd.read_csv(args.only_ids_from, low_memory=False)
        if "ok" in ids.columns:
            ids = ids.loc[true_mask(ids["ok"])]
        keep = set(ids["song_id"].astype(str).str.strip())
        merged = merged[merged["song_id"].isin(keep)].reset_index(drop=True)
        print(f"Restricted to {len(merged)} tracks listed in {args.only_ids_from}")
    y = (merged["source"] == "AI").astype(int).values  # 1 = AI
    groups = merged["group_id"].values
    split = merged["holdout_split"].values
    tr, va, te = (np.where(split == s)[0] for s in ("train", "val", "test"))
    print(f"\nAudio features: {args.audio} | tracks: {len(merged)} | train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print("Features:", {k: len(v) for k, v in sets.items()})
    assert not set(groups[tr]) & set(groups[te]) and not set(groups[tr]) & set(groups[va]) \
        and not set(groups[va]) & set(groups[te]), "a group crosses splits"

    base, grids = build_models(args.quick)
    if args.models:
        keep = {m.strip() for m in args.models.split(",")}
        base = {k: v for k, v in base.items() if k in keep}
    wanted = [s.strip() for s in args.sets.split(",") if s.strip()]
    # small subsets (for example a Colab run with --per-cell) need fewer inner folds
    min_class = int(min(np.bincount(y[tr], minlength=2))) if len(tr) else 0
    min_groups = min(len(set(groups[tr][y[tr] == c])) for c in (0, 1)) if len(tr) else 0
    n_inner = min(5, min_class, min_groups)
    if n_inner < 2:
        sys.exit(f"STOP: only {len(tr)} training tracks (smallest class {min_class}, {min_groups} groups); "
                 "too few for cross-validation. Use a larger --per-cell value in the harmonise/extract steps.")
    if n_inner < 5:
        print(f"NOTE: small subset, inner CV uses {n_inner} folds instead of 5.")
    if len(merged) < 300:
        print(f"NOTE: only {len(merged)} tracks; results are for a pipeline check, not for the paper.")
    inner = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=RANDOM_SEED)

    rows, fitted, best_params = [], {}, {}
    for fs in wanted:
        cols = sets[fs]
        X = merged[cols].apply(pd.to_numeric, errors="coerce")
        print(f"\n=== {fs} ({len(cols)} features) ===", flush=True)
        for name, pipe in base.items():
            t0 = time.time()
            search = GridSearchCV(clone(pipe), grids[name], scoring="f1_macro", cv=inner, n_jobs=args.jobs,
                                  refit=True, error_score="raise")
            # StratifiedGroupKFold stratifies on y; groups keep artists / duplicate lyrics together
            search.fit(X.iloc[tr], y[tr], groups=groups[tr])
            m = search.best_estimator_
            fitted[(fs, name)] = m
            best_params[(fs, name)] = search.best_params_
            for sp_name, idx in (("validation", va), ("test", te)):
                rows.append({"feature_set": fs, "model": name, "split": sp_name, "n": len(idx),
                             "cv_f1": search.best_score_, **metrics(m, X.iloc[idx], y[idx])})
            print(f"  {name:<20} val F1={rows[-2]['f1_macro']:.4f} | test F1={rows[-1]['f1_macro']:.4f} "
                  f"AUROC={rows[-1]['auroc']:.4f} | {search.best_params_} | {(time.time() - t0) / 60:.1f} min",
                  flush=True)

    res = pd.DataFrame(rows).round(4)
    res.to_csv(REVISION_DIR / f"grouped_{tag}_all_models.csv", index=False, encoding="utf-8-sig")

    # ---- validation-selected model per feature set (ties broken by validation AUROC) ----
    val = res[res["split"] == "validation"].sort_values(["feature_set", "f1_macro", "auroc"],
                                                        ascending=[True, False, False])
    chosen = val.groupby("feature_set").head(1)[["feature_set", "model"]]
    ties = val.groupby("feature_set").apply(
        lambda d: d[d["f1_macro"] >= d["f1_macro"].max() - 1e-4]["model"].tolist())
    summary = []
    for _, r in chosen.iterrows():
        fs, name = r["feature_set"], r["model"]
        m = fitted[(fs, name)]
        X = merged[sets[fs]].apply(pd.to_numeric, errors="coerce")
        pred = m.predict(X.iloc[te])
        sc = scores(m, X.iloc[te])
        (f_lo, f_hi), (a_lo, a_hi) = group_bootstrap(y[te], pred, sc, groups[te])
        mt = metrics(m, X.iloc[te], y[te])
        # per-language test F1
        per_lang = {lang: round(f1_score(y[te][merged["language"].values[te] == lang],
                                          pred[merged["language"].values[te] == lang], average="macro"), 4)
                    for lang in sorted(set(merged["language"]))}
        row = {"feature_set": fs, "model": name, "test_f1": round(mt["f1_macro"], 4),
               "f1_ci_low": round(f_lo, 4), "f1_ci_high": round(f_hi, 4), "test_auroc": round(mt["auroc"], 4),
               "auc_ci_low": round(a_lo, 4), "auc_ci_high": round(a_hi, 4),
               "tied_on_validation": "|".join(ties[fs]), "paper_f1_random_split": PAPER.get(fs, np.nan),
               **{f"f1_{k}": v for k, v in per_lang.items()}}
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary["change_vs_paper"] = (summary["test_f1"] - summary["paper_f1_random_split"]).round(4)

    # ---- file-format-only baseline ----
    fb = format_baseline(merged, y, tr, te)
    if fb:
        summary = pd.concat([summary, pd.DataFrame([{"feature_set": "FORMAT ONLY (codec, bitrate, ...)",
                                                       "model": "LogReg", "test_f1": round(fb["f1_macro"], 4),
                                                       "test_auroc": round(fb["auroc"], 4)}])], ignore_index=True)
    summary.to_csv(REVISION_DIR / f"grouped_{tag}_summary.csv", index=False, encoding="utf-8-sig")

    # ---- shuffled-label control: permute the TRAINING labels several times, report mean +/- sd ----
    # One permutation can by chance agree with the true labels on 52-53% of training rows; with features this
    # strong that faint real signal gets learned and F1 lands above 0.5 without any leakage. Many permutations
    # average this out (expected mean = 0.5).
    if "fused" in wanted:
        print(f"\n=== Shuffled-label control ({args.shuffles} permutations of the training labels, fused features) ===")
        Xf = merged[sets["fused"]].apply(pd.to_numeric, errors="coerce")
        ctrl = []
        for name, pipe in base.items():
            f1s, agree = [], []
            for r in range(args.shuffles):
                y_s = np.random.RandomState(RANDOM_SEED + r).permutation(y[tr])
                agree.append(float(np.mean(y_s == y[tr])))
                mdl = clone(pipe).fit(Xf.iloc[tr], y_s)
                f1s.append(f1_score(y[te], mdl.predict(Xf.iloc[te]), average="macro"))
            row = {"model": name, "mean_f1": round(float(np.mean(f1s)), 4), "sd_f1": round(float(np.std(f1s)), 4),
                   "min_f1": round(float(np.min(f1s)), 4), "max_f1": round(float(np.max(f1s)), 4),
                   "mean_label_agreement_with_truth": round(float(np.mean(agree)), 4),
                   "status": "OK" if np.mean(f1s) < 0.55 else "CHECK"}
            ctrl.append(row)
            print(f"  {name:<20} mean F1={row['mean_f1']:.4f} sd={row['sd_f1']:.4f} "
                  f"(range {row['min_f1']:.3f}-{row['max_f1']:.3f}) {row['status']}")
        pd.DataFrame(ctrl).to_csv(REVISION_DIR / f"grouped_{tag}_shuffled_control.csv", index=False)

    # ---- repeated grouped CV with the tuned settings fixed (mean +/- std over folds) ----
    if not args.no_cv and any(c.endswith("_fold") for c in merged.columns):
        print("\n=== Repeated grouped 5-fold CV (tuned settings fixed) ===")
        cv_rows = []
        reps = sorted(c for c in merged.columns if c.endswith("_fold"))
        for _, r in chosen.iterrows():
            fs, name = r["feature_set"], r["model"]
            X = merged[sets[fs]].apply(pd.to_numeric, errors="coerce")
            f1s, aucs = [], []
            for rc in reps:
                folds = merged[rc].values
                for k in np.unique(folds):
                    trn, tst = np.where(folds != k)[0], np.where(folds == k)[0]
                    mdl = clone(fitted[(fs, name)]).fit(X.iloc[trn], y[trn])
                    mt = metrics(mdl, X.iloc[tst], y[tst])
                    f1s.append(mt["f1_macro"])
                    aucs.append(mt["auroc"])
            cv_rows.append({"feature_set": fs, "model": name, "folds": len(f1s),
                            "f1_mean": round(np.mean(f1s), 4), "f1_std": round(np.std(f1s), 4),
                            "auroc_mean": round(np.nanmean(aucs), 4), "auroc_std": round(np.nanstd(aucs), 4)})
            print(f"  {fs:<18} {name:<20} F1 = {np.mean(f1s):.4f} +/- {np.std(f1s):.4f} ({len(f1s)} folds)",
                  flush=True)
        pd.DataFrame(cv_rows).to_csv(REVISION_DIR / f"grouped_{tag}_repeated_cv.csv", index=False)

    json.dump({f"{k[0]}|{k[1]}": {a: str(b) for a, b in v.items()} for k, v in best_params.items()},
              open(REVISION_DIR / f"grouped_{tag}_best_params.json", "w"), indent=1)

    print("\n" + "=" * 78)
    print(f"GROUPED-SPLIT RESULTS  (audio = {args.audio}, test N = {len(te)}; no artist/lyric group shared with train)")
    print("=" * 78)
    show = [c for c in ["feature_set", "model", "test_f1", "f1_ci_low", "f1_ci_high", "test_auroc",
                        "paper_f1_random_split", "change_vs_paper", "tied_on_validation"] if c in summary.columns]
    print(summary[show].to_string(index=False))
    print("\nSaved to", REVISION_DIR, f"(grouped_{tag}_*.csv)")


if __name__ == "__main__":
    main()
