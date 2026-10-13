"""
Cross-generator test: train on your corpus (Suno v5 AI vs human),
then test on AI songs from a generator the model never saw.

Three sources of unseen AI songs (pick one per run; results are saved separately):
  --generator udio    Udio songs from the SONICS dataset (default)
  --generator suno    older Suno versions from SONICS (the model saw Suno v5 only)
                      optional: --algorithm v3   keeps only versions whose name contains "v3"
  --own-dir FOLDER --own-name NAME
                      your own files, e.g. tracks you generated with a newer Suno model or another
                      generator. Put the audio files (mp3 / wav / m4a / flac) in FOLDER.

What it does
  1. Downloads the SONICS metadata and N Udio songs from Hugging Face (only those files,
     not the 32 GB dataset). Songs: fully AI ("full fake"), with vocals, >= 60 s, one per lyric.
  2. Extracts exactly the same audio features as your notebook (parallel).
  3. Trains the audio-only model the same way as the fusion notebook (LR and SVM,
     grid search, choice by validation F1), using the grouped split from make_grouped_splits.py.
  4. Tests it on: held-out human songs + held-out Suno songs  (in-distribution)
                  held-out human songs + Udio songs           (new generator)
  5. Checks whether the Udio files also differ from the human files in file format.

Run from the project folder (in Jupyter use the run() helper):
    run("cross_generator_test.py", "--n", "300")
    run("cross_generator_test.py", "--generator", "suno", "--n", "100")
    run("cross_generator_test.py", "--own-dir", r"D:\\new_suno", "--own-name", "suno_new")
If you downloaded SONICS yourself (e.g. from Kaggle), point to that folder instead:
    run("cross_generator_test.py", "--n", "300", "--local-dir", r"D:\\sonics\\dataset")
Everything resumes if interrupted. Dataset licence: CC BY-NC 4.0 (research use).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import shutil
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from common import AUDIO_CSV, REVISION_DIR, RANDOM_SEED, true_mask
from extract_harmonised_features import extract_all
from audit_dataset import ffprobe

warnings.filterwarnings("ignore")
REPO = "awsaf49/sonics"
HUB_BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
SONICS_DIR = Path("sonics_download")
ZIP_INDEX = REVISION_DIR / "sonics_zip_index.csv"
UDIO_MANIFEST = REVISION_DIR / "udio_manifest.csv"          # re-set in main() for other generators
UDIO_FEATURES = REVISION_DIR / "udio_audio_features.csv"
AUDIO_EXT = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus"}

# Same exclusions as 03_multimodal_fusion_final.ipynb
META = {"song_id", "title", "artist", "language", "final_genre", "source", "audio_path",
        "original_audio_path", "ok", "error", "is_mono", "is_clipped", "duration_seconds",
        "duration_sec", "label", "target"}
AUDIO_EXCLUDE = {"lufs_integrated", "rms_db", "sample_peak_dbfs", "true_peak_dbfs",
                 "hard_clip_count", "clip_rate_per_sec", "is_clipped", "silence_ratio",
                 "pentatonic_energy_ratio", "pitch_dev_mean_cents", "pitch_dev_std_cents",
                 "pitch_microtonal_ratio", "tuning_offset_cents", "beat_regularity_cv",
                 "mfcc_1_mean", "mfcc_1_std"}


# ------------------------------------------------------------------ 1. select + download
def select_udio(meta, n, seed, generator="udio", algorithm=""):
    m = meta.copy()
    m = m[m["source"].astype(str).str.lower().eq(generator)]
    print(f"{generator} songs in SONICS metadata: {len(m)}")
    if "algorithm" in m.columns:
        print("  versions available:", m["algorithm"].astype(str).value_counts().to_dict())
        if algorithm:
            m = m[m["algorithm"].astype(str).str.lower().str.contains(algorithm.lower(), regex=False)]
            print(f"  after --algorithm {algorithm!r}: {len(m)}")
    elif algorithm:
        print("  WARNING: no 'algorithm' column in the metadata; --algorithm ignored")
    if "label" in m.columns:
        m = m[m["label"].astype(str).str.lower().eq("full fake")]
    if "no_vocal" in m.columns:
        m = m[~m["no_vocal"].astype(str).str.lower().eq("true")]
    if "duration" in m.columns:
        m = m[pd.to_numeric(m["duration"], errors="coerce") >= 60]
    if "id" in m.columns:                       # two versions share one id (same lyrics) -> keep one
        m = m.drop_duplicates("id")
    print(f"After filters (full fake, vocals, >= 60 s, one per lyric): {len(m)}")
    if "split" in m.columns and (m["split"] == "test").sum() >= n:
        m = m[m["split"] == "test"]             # prefer SONICS's own test split
    # fixed shuffled order, so a small test run (--n 10) is the start of the full run (--n 300)
    return m.sample(frac=1, random_state=seed).head(n)


def file_name(row):
    raw = str(row.get("filepath") if pd.notna(row.get("filepath", np.nan)) else row.get("filename"))
    name = Path(raw).name
    return name if name.lower().endswith(".mp3") else name + ".mp3"


def resolve_url(path):
    """Hugging Face answers with a redirect to a temporary download link; follow it once."""
    import requests
    r = requests.get(HUB_BASE + path, headers={"Range": "bytes=0-0"}, stream=True,
                     allow_redirects=True, timeout=60)
    r.close()
    if r.status_code not in (200, 206):
        raise SystemExit(f"Could not open {path} (HTTP {r.status_code})")
    return r.url


def need_remotezip():
    try:
        from remotezip import RemoteZip
        return RemoteZip
    except ImportError:
        raise SystemExit("Please run:  pip install remotezip   and try again.")


def zip_index(zips):
    """Which zip part holds which song. Reads only each zip's table of contents, not the audio."""
    if ZIP_INDEX.exists():
        idx = pd.read_csv(ZIP_INDEX)
        if set(zips) <= set(idx["part"]):
            return idx
    RemoteZip = need_remotezip()
    rows = []
    for part in sorted(zips):
        print(f"  reading table of contents of {part} ...", flush=True)
        with RemoteZip(resolve_url(part)) as z:
            for info in z.infolist():
                if info.filename.lower().endswith(".mp3"):
                    rows.append({"file": Path(info.filename).name, "part": part, "member": info.filename})
    idx = pd.DataFrame(rows)
    idx.to_csv(ZIP_INDEX, index=False)
    print(f"  {len(idx)} songs found in {len(zips)} zip files (saved {ZIP_INDEX.name})")
    return idx


def download_from_zips(chosen, idx, outdir):
    """Pull out only the chosen songs from the big zip files (HTTP range requests)."""
    RemoteZip = need_remotezip()
    outdir.mkdir(parents=True, exist_ok=True)
    where = idx.drop_duplicates("file").set_index("file")
    todo = [(f, where.loc[f, "part"], where.loc[f, "member"]) for f in chosen["file"]
            if not (outdir / f).exists()]
    print(f"Downloading {len(todo)} songs from the zip files ({len(chosen) - len(todo)} already on disk)...")
    done = 0
    for part in sorted({t[1] for t in todo}):
        items = [t for t in todo if t[1] == part]
        with RemoteZip(resolve_url(part)) as z:
            for f, _, member in items:
                tmp = outdir / (f + ".part")
                with z.open(member) as src, open(tmp, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                tmp.replace(outdir / f)
                done += 1
                if done % 20 == 0 or done == len(todo):
                    print(f"  {done}/{len(todo)} downloaded", flush=True)
    return [str(outdir / f) for f in chosen["file"]]


def prepare_from_hub(n, seed, generator="udio", algorithm=""):
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        raise SystemExit("Please run:  pip install huggingface_hub   and try again.")
    print("Listing SONICS files on Hugging Face (can take a minute)...")
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    csvs = [f for f in files if f.lower().endswith(".csv")]
    mp3s = {Path(f).name: f for f in files if f.lower().endswith(".mp3")}
    print(f"  CSV files: {csvs}")
    print(f"  MP3 files available individually: {len(mp3s)}")
    zips = [f for f in files if f.lower().endswith(".zip") and "fake" in f.lower()]
    if not mp3s and not zips:
        others = [f for f in files if not f.lower().endswith(".csv")][:20]
        raise SystemExit("The audio is not stored as single MP3 files or zip parts. Please send me this list:\n"
                         + "\n".join(others))
    if zips:
        print(f"  Audio is stored in {len(zips)} zip files; only the songs needed will be downloaded.")
    fake_csv = next((f for f in csvs if "fake" in Path(f).name.lower()), None)
    if fake_csv is None:
        raise SystemExit(f"No fake_songs CSV found. CSV files: {csvs}")
    meta = pd.read_csv(hf_hub_download(REPO, fake_csv, repo_type="dataset", local_dir=SONICS_DIR),
                       low_memory=False)
    if zips and not mp3s:
        idx = zip_index(zips)
        chosen = select_udio(meta, n * 2, seed, generator, algorithm)   # spare rows in case a file is missing
        chosen["file"] = chosen.apply(file_name, axis=1)
        chosen = chosen[chosen["file"].isin(set(idx["file"]))].head(n).copy()
        if chosen.empty:
            raise SystemExit("None of the selected songs were found in the zip files.")
        chosen["audio_path"] = download_from_zips(chosen, idx, SONICS_DIR / "fake_songs")
        return chosen
    chosen = select_udio(meta, n, seed, generator, algorithm)
    chosen["file"] = chosen.apply(file_name, axis=1)
    chosen = chosen[chosen["file"].isin(mp3s)]
    print(f"Downloading {len(chosen)} {generator} songs (skips files already downloaded)...")

    def get(name):
        target = SONICS_DIR / mp3s[name]
        if not target.exists():
            hf_hub_download(REPO, mp3s[name], repo_type="dataset", local_dir=SONICS_DIR)
        return str(target)

    with ThreadPoolExecutor(max_workers=6) as pool:
        chosen["audio_path"] = list(pool.map(get, chosen["file"]))
    return chosen


def prepare_from_local(local_dir, n, seed, generator="udio", algorithm=""):
    local_dir = Path(local_dir)
    csv = next((p for p in local_dir.rglob("*.csv") if "fake" in p.name.lower()), None)
    if csv is None:
        raise SystemExit(f"No fake_songs CSV found under {local_dir}")
    meta = pd.read_csv(csv, low_memory=False)
    index = {p.name: p for p in local_dir.rglob("*.mp3")}
    print(f"MP3 files found on disk: {len(index)}")
    meta = meta[meta.apply(file_name, axis=1).isin(index)]      # only songs you actually have
    chosen = select_udio(meta, n, seed, generator, algorithm)
    chosen["file"] = chosen.apply(file_name, axis=1)
    chosen["audio_path"] = [str(index[f]) if f in index else None for f in chosen["file"]]
    missing = chosen["audio_path"].isna().sum()
    if missing:
        print(f"WARNING: {missing} selected songs not found on disk - they are skipped")
    return chosen.dropna(subset=["audio_path"])


def prepare_from_own(own_dir, own_name):
    own_dir = Path(own_dir)
    files = sorted(p for p in own_dir.rglob("*") if p.suffix.lower() in AUDIO_EXT)
    if not files:
        raise SystemExit(f"No audio files ({', '.join(sorted(AUDIO_EXT))}) found under {own_dir}")
    print(f"Found {len(files)} audio files in {own_dir}")
    return pd.DataFrame({
        "song_id": [f"{own_name.upper()}_{p.stem}" for p in files],
        "audio_path": [str(p) for p in files],
        "sonics_genre": "", "sonics_split": "", "duration": np.nan,
    })


# ------------------------------------------------------------------ 2. features
def extract_udio(manifest, workers):
    done = pd.read_csv(UDIO_FEATURES, low_memory=False) if UDIO_FEATURES.exists() else pd.DataFrame()
    finished = set(done.loc[true_mask(done["ok"]), "song_id"]) if len(done) else set()
    todo = manifest[~manifest["song_id"].isin(finished)]
    print(f"\nExtracting features: {len(todo)} to do, {len(finished)} already done, {workers} workers")
    records = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(extract_all, r.song_id, r.audio_path) for r in todo.itertuples(index=False)]
        for n, fut in enumerate(as_completed(futures), 1):
            records.append(fut.result())
            if n % 25 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} done")
                pd.concat([done, pd.DataFrame(records)]).drop_duplicates("song_id", keep="last") \
                  .to_csv(UDIO_FEATURES, index=False, encoding="utf-8-sig")
    feats = pd.read_csv(UDIO_FEATURES, low_memory=False)
    feats = feats.loc[true_mask(feats["ok"])]
    print(f"Tracks with features: {len(feats)}")
    return feats


# ------------------------------------------------------------------ 3. train like the notebook
def numeric_features(df):
    cols = []
    for col in df.columns:
        if col in META or col in AUDIO_EXCLUDE or col.startswith("muq_"):
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() == 0 or values.isna().mean() >= 0.50:
            continue
        cols.append(col)
    return cols


def make_pipe(clf):
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("variance", VarianceThreshold(0.0)),
                     ("scaler", StandardScaler()), ("clf", clf)])


MODELS = {
    "Logistic Regression": (make_pipe(LogisticRegression(max_iter=2000, class_weight="balanced",
                                                         random_state=RANDOM_SEED)),
                            {"clf__C": [0.01, 0.1, 1.0, 10.0]}),
    "SVM (RBF)": (make_pipe(SVC(probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
                  {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]}),
}


def train_audio_model(corpus, feat_cols):
    tr = corpus[corpus["holdout_split"] == "train"]
    va = corpus[corpus["holdout_split"] == "val"]
    Xtr, ytr = tr[feat_cols].apply(pd.to_numeric, errors="coerce"), (tr["source"] == "AI").astype(int)
    Xva, yva = va[feat_cols].apply(pd.to_numeric, errors="coerce"), (va["source"] == "AI").astype(int)
    best = (None, -1, None)
    for name, (pipe, grid) in MODELS.items():
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
        search = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1)
        search.fit(Xtr, ytr, groups=tr["group_id"])
        val_f1 = f1_score(yva, search.predict(Xva), average="macro")
        print(f"  {name:<20} best {search.best_params_}  validation F1 = {val_f1:.4f}")
        if val_f1 > best[1]:
            best = (name, val_f1, search.best_estimator_)
    print(f"  Selected on validation: {best[0]}")
    return best[0], best[2]


def evaluate(model, feat_cols, humans, ai_sets):
    rows = []
    Xh = humans[feat_cols].apply(pd.to_numeric, errors="coerce")
    ph = model.predict_proba(Xh)[:, 1]
    for label, ai in ai_sets.items():
        Xa = ai[feat_cols].apply(pd.to_numeric, errors="coerce")
        pa = model.predict_proba(Xa)[:, 1]
        y = np.r_[np.zeros(len(ph)), np.ones(len(pa))]
        p = np.r_[ph, pa]
        pred = (p >= 0.5).astype(int)
        rows.append({"test": label, "n_human": len(ph), "n_ai": len(pa),
                     "AUROC": round(roc_auc_score(y, p), 4),
                     "AI_detected_%": round(100 * (pa >= 0.5).mean(), 1),
                     "Human_correct_%": round(100 * (ph < 0.5).mean(), 1),
                     "macro_F1": round(f1_score(y, pred, average="macro"), 4)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="number of songs to test (SONICS modes)")
    ap.add_argument("--generator", choices=["udio", "suno"], default="udio",
                    help="which SONICS generator to use as the unseen AI source")
    ap.add_argument("--algorithm", default="", help="keep only SONICS versions containing this text, e.g. v3")
    ap.add_argument("--own-dir", default="", help="folder with your own AI-generated files (instead of SONICS)")
    ap.add_argument("--own-name", default="own", help="short name for your own files, e.g. suno_new")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--local-dir", default="", help="use an already downloaded SONICS folder")
    args = ap.parse_args()

    global UDIO_MANIFEST, UDIO_FEATURES
    if args.own_dir:
        tag = "".join(c if c.isalnum() else "_" for c in args.own_name.lower())
        gen_name = args.own_name
    else:
        tag = args.generator + (("_" + "".join(c if c.isalnum() else "_" for c in args.algorithm.lower()))
                                if args.algorithm else "")
        gen_name = args.generator.capitalize() + (f" ({args.algorithm})" if args.algorithm else "")
    UDIO_MANIFEST = REVISION_DIR / f"{tag}_manifest.csv"
    UDIO_FEATURES = REVISION_DIR / f"{tag}_audio_features.csv"

    # 1. unseen-generator songs
    existing = pd.read_csv(UDIO_MANIFEST) if UDIO_MANIFEST.exists() else pd.DataFrame()
    if args.own_dir:
        manifest = prepare_from_own(args.own_dir, args.own_name)
        manifest.to_csv(UDIO_MANIFEST, index=False, encoding="utf-8-sig")
    elif len(existing) >= args.n:
        manifest = existing.head(args.n)
        print(f"Using existing {gen_name} list: {len(manifest)} songs ({UDIO_MANIFEST})")
    else:
        chosen = (prepare_from_local(args.local_dir, args.n, RANDOM_SEED, args.generator, args.algorithm)
                  if args.local_dir else prepare_from_hub(args.n, RANDOM_SEED, args.generator, args.algorithm))
        manifest = pd.DataFrame({
            "song_id": [args.generator.upper() + "_" + Path(f).stem for f in chosen["file"]],
            "audio_path": chosen["audio_path"].values,
            "sonics_genre": chosen["genre"].values if "genre" in chosen else "",
            "sonics_split": chosen["split"].values if "split" in chosen else "",
            "duration": chosen["duration"].values if "duration" in chosen else np.nan,
        })
        manifest.to_csv(UDIO_MANIFEST, index=False, encoding="utf-8-sig")
        print(f"Saved {gen_name} list: {UDIO_MANIFEST}")

    # 2. features
    udio = extract_udio(manifest, args.workers)
    udio = udio[udio["song_id"].isin(manifest["song_id"])]

    # 3. corpus + grouped split
    corpus = pd.read_csv(AUDIO_CSV, low_memory=False)
    corpus["song_id"] = corpus["song_id"].astype(str).str.strip()
    corpus = corpus.loc[true_mask(corpus["ok"])].drop_duplicates("song_id", keep="last")
    splits_path = REVISION_DIR / "splits.csv"
    if not splits_path.exists():
        raise SystemExit("outputs/revision/splits.csv not found - run make_grouped_splits.py first.")
    splits = pd.read_csv(splits_path)[["song_id", "holdout_split", "group_id"]]
    corpus = corpus.merge(splits, on="song_id", how="inner")
    feat_cols = [c for c in numeric_features(corpus) if c in udio.columns]
    print(f"\nAudio features used: {len(feat_cols)} (paper: 78)")
    print("Training the audio-only model (grouped split, same settings as the notebook)...")
    name, model = train_audio_model(corpus, feat_cols)

    # 4. evaluation
    test = corpus[corpus["holdout_split"] == "test"]
    humans, suno = test[test["source"] == "Human"], test[test["source"] == "AI"]
    res_all = evaluate(model, feat_cols, humans,
                       {"Suno v5 test songs (seen generator)": suno, f"{gen_name} songs (new generator)": udio})
    en = test["language"] == "English"
    res_en = evaluate(model, feat_cols, test[en & (test["source"] == "Human")],
                      {"EN: Suno v5 test songs": test[en & (test["source"] == "AI")],
                       f"EN: {gen_name} songs": udio})
    results = pd.concat([res_all, res_en], ignore_index=True)
    results.insert(0, "model", name)
    results.to_csv(REVISION_DIR / f"{tag}_cross_generator_results.csv", index=False, encoding="utf-8-sig")
    print("\n=== Cross-generator results (same held-out human songs in every row) ===")
    print(results.to_string(index=False))

    # 5. file-format check
    print(f"\n=== File format: {gen_name} vs human test files ===")
    with ThreadPoolExecutor(max_workers=8) as pool:
        fmt_u = pd.DataFrame(list(pool.map(ffprobe, udio["audio_path"].astype(str))))
        fmt_h = pd.DataFrame(list(pool.map(ffprobe, humans["audio_path"].astype(str))))
    for label, fmt in [(gen_name[:10], fmt_u), ("Human", fmt_h)]:
        if "bitrate_kbps" in fmt:
            print(f"  {label:<6} codec {fmt['codec'].value_counts().to_dict()} | "
                  f"sample rate {fmt['sample_rate'].value_counts().to_dict()} | "
                  f"bitrate kbps median {fmt['bitrate_kbps'].median():.0f} "
                  f"(range {fmt['bitrate_kbps'].min():.0f}-{fmt['bitrate_kbps'].max():.0f})")
    print("  If this generator's format also differs from the human format, a high score could still")
    print("  partly reflect file format. State this in the paper, or re-test on harmonised audio later.")
    print(f"\nSaved: {REVISION_DIR / (tag + '_cross_generator_results.csv')}")


if __name__ == "__main__":
    main()
