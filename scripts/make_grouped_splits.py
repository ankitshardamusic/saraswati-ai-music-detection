"""
Step 2: build leakage-safe groups and grouped splits.

Groups (songs that must never be split across train and test):
  1. Human songs by the same primary artist.
  2. Near-duplicate lyrics within a language. This catches Suno's two variations of
     one prompt (same lyrics, different audio), Muse tracks that share a song_id,
     and the same human song uploaded twice.
  3. Muse `song_id_original`, if an old Muse manifest with that column still exists.

Outputs (outputs/revision/):
  groups.csv                  song_id -> group_id, with the reason each link was made
  near_duplicate_pairs.csv    every lyric pair above threshold, to check by eye
  leakage_original_split.csv  how many ORIGINAL test tracks shared a group with training
  splits.csv                  grouped 60/20/20 split + 5 repeats x 5 grouped folds

Run from the project folder:
    python make_grouped_splits.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import StratifiedGroupKFold

from common import (DATA_DIR, TRANSCRIPTS_DIR, REVISION_DIR, load_embeddings,
                          load_merged, preflight, primary_artist, reproduce_original_split)

LYRIC_JACCARD_THRESHOLD = 0.30  # overlap of word 3-grams (character 3-grams for Mandarin).
                                 # Unrelated songs score close to 0; the same lyrics sung twice
                                 # score far higher. Check near_duplicate_pairs.csv and adjust.
EMB_SIM_THRESHOLD = 0.97         # sentence-embedding cosine; second, independent signal
MIN_TRANSCRIPT_CHARS = 40    # very short transcripts (instrumental/failed ASR) are never linked
MIN_UNIQUE_SHINGLES = 15     # transcripts with fewer distinct 3-grams are degenerate (e.g. Whisper
                             # repeating one word) and are never linked - they are not real lyrics
N_REPEATS, N_FOLDS = 5, 5


class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}
        self.reason = {i: set() for i in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b, why):
        ra, rb = self.find(a), self.find(b)
        self.reason[a].add(why)
        self.reason[b].add(why)
        if ra != rb:
            self.parent[rb] = ra


def read_transcript(song_id):
    path = TRANSCRIPTS_DIR / f"{song_id}.txt"
    return path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""


_WORD = re.compile(r"[a-z0-9'\u00c0-\u024f\u0900-\u097f\u0600-\u06ff\u0750-\u077f]+")
_HAN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]")


def shingles(text, language, n=3):
    text = str(text).casefold()
    han = _HAN.findall(text)
    if language == "Chinese" and len(han) >= 30:
        units = han                                 # characters play the role of words
    else:
        units = _WORD.findall(text)                 # also covers English songs in the Mandarin set
    return {" ".join(units[i:i + n]) for i in range(len(units) - n + 1)}


def remove_boilerplate_lines(texts, merged):
    cleaned = dict(texts)
    for lang, part in merged.groupby("language"):
        sids = part["song_id"].tolist()
        doc_freq = {}
        for sid in sids:
            for line in {l.strip().casefold() for l in texts[sid].splitlines() if l.strip()}:
                doc_freq[line] = doc_freq.get(line, 0) + 1
        limit = max(3, int(0.01 * len(sids)))
        common = {line for line, n in doc_freq.items() if n > limit}
        if common:
            print(f"  {lang}: removed {len(common)} boilerplate lines seen in >{limit} transcripts, e.g. "
                  f"{sorted(common, key=lambda l: -doc_freq[l])[:3]}")
        for sid in sids:
            cleaned[sid] = "\n".join(l for l in texts[sid].splitlines()
                                     if l.strip() and l.strip().casefold() not in common)
    return cleaned


def find_muse_ids():
    """Look for any surviving manifest that still has Muse's original song_id."""
    found = {}
    for csv in DATA_DIR.rglob("*.csv"):
        try:
            head = pd.read_csv(csv, nrows=2)
        except Exception:
            continue
        if {"song_id_original", "new_song_id"}.issubset(head.columns):
            df = pd.read_csv(csv, usecols=["song_id_original", "new_song_id"])
            for new_id, orig in zip(df["new_song_id"].astype(str), df["song_id_original"].astype(str)):
                found[new_id.strip()] = orig.strip()
            print(f"  Muse IDs found in {csv} ({len(df)} rows)")
    return found


def build_groups(merged):
    ids = merged["song_id"].tolist()
    uf = UnionFind(ids)
    edges = []  # (song_a, song_b, reason) - used for the per-reason leakage report

    # 1. Human primary artist
    merged["primary_artist"] = [primary_artist(a) if s == "Human" else ""
                                for a, s in zip(merged["artist"], merged["source"])]
    for artist, part in merged[merged["primary_artist"] != ""].groupby("primary_artist"):
        members = part["song_id"].tolist()
        for n, a in enumerate(members):
            for b in members[n + 1:]:
                edges.append((a, b, "same_artist"))
        for other in members[1:]:
            uf.union(members[0], other, "same_artist")

    # 2a. Near-duplicate lyrics (character n-grams work for all three scripts).
    # Lines that appear in many transcripts (Whisper hallucinations such as
    # "thanks for watching" or subtitle credits) are removed first, otherwise they
    # would link unrelated songs.
    texts = {sid: read_transcript(sid) for sid in ids}
    texts = remove_boilerplate_lines(texts, merged)
    lang_of_id = merged.set_index("song_id")["language"].to_dict()
    n_unique = {sid: len(shingles(texts[sid], lang_of_id[sid])) for sid in ids}
    usable = {sid for sid in ids
              if len(texts[sid].strip()) >= MIN_TRANSCRIPT_CHARS and n_unique[sid] >= MIN_UNIQUE_SHINGLES}
    merged["transcript_unique_3grams"] = merged["song_id"].map(n_unique)
    merged["transcript_usable"] = merged["song_id"].isin(usable)
    pair_rows = []
    for lang, part in merged.groupby("language"):
        sids = [s for s in part["song_id"] if s in usable]
        if len(sids) < 2:
            continue
        vec = CountVectorizer(analyzer=lambda t, L=lang: shingles(t, L), binary=True)
        X = vec.fit_transform([texts[s] for s in sids]).astype(np.float32)
        inter = (X @ X.T).toarray()
        size = np.asarray(X.sum(axis=1)).ravel()
        jac = inter / (size[:, None] + size[None, :] - inter + 1e-9)
        np.fill_diagonal(jac, 0)
        src = part.set_index("song_id").loc[sids, "source"].values
        for i, j in zip(*np.where(np.triu(jac) >= LYRIC_JACCARD_THRESHOLD / 2)):
            pair_rows.append({"song_a": sids[i], "song_b": sids[j], "language": lang,
                              "source_a": src[i], "source_b": src[j],
                              "lyric_jaccard": round(float(jac[i, j]), 4),
                              "linked": bool(jac[i, j] >= LYRIC_JACCARD_THRESHOLD)})
            if jac[i, j] >= LYRIC_JACCARD_THRESHOLD:
                uf.union(sids[i], sids[j], "near_duplicate_lyrics")
                edges.append((sids[i], sids[j], "near_duplicate_lyrics"))

    # 2b. Near-duplicate sentence embeddings (independent check)
    emb = load_embeddings().set_index("song_id")
    emb = emb.loc[[s for s in ids if s in emb.index and s in usable]]
    E = emb.values / (np.linalg.norm(emb.values, axis=1, keepdims=True) + 1e-12)
    lang_of = merged.set_index("song_id")["language"]
    esim = E @ E.T
    np.fill_diagonal(esim, 0)
    emb_ids = emb.index.tolist()
    for i, j in zip(*np.where(np.triu(esim) >= EMB_SIM_THRESHOLD)):
        if lang_of[emb_ids[i]] == lang_of[emb_ids[j]]:
            uf.union(emb_ids[i], emb_ids[j], "near_duplicate_embedding")
            edges.append((emb_ids[i], emb_ids[j], "near_duplicate_embedding"))

    # 3. Muse original song IDs
    muse = find_muse_ids()
    by_orig = {}
    for sid in ids:
        if sid in muse:
            by_orig.setdefault(muse[sid], []).append(sid)
    for members in by_orig.values():
        for other in members[1:]:
            uf.union(members[0], other, "same_muse_song")
            edges.append((members[0], other, "same_muse_song"))

    roots = {sid: uf.find(sid) for sid in ids}
    root_to_gid = {r: f"G{n:04d}" for n, r in enumerate(sorted(set(roots.values())))}
    merged["group_id"] = [root_to_gid[roots[s]] for s in ids]
    merged["link_reasons"] = [";".join(sorted(uf.reason[s])) or "singleton" for s in ids]

    pairs = pd.DataFrame(pair_rows).sort_values("lyric_jaccard", ascending=False) if pair_rows else pd.DataFrame()
    return merged, pairs, edges


def report_groups(merged, pairs):
    bad = merged[~merged["transcript_usable"]]
    print("\n=== Transcripts too short or degenerate to count as lyrics (not used for linking) ===")
    print(pd.crosstab(bad["language"], bad["source"]).to_string() if len(bad) else "None")
    bad[["song_id", "source", "language", "final_genre", "transcript_unique_3grams"]] \
        .to_csv(REVISION_DIR / "degenerate_transcripts.csv", index=False, encoding="utf-8-sig")
    sizes = merged.groupby("group_id").size()
    print("\n=== Groups ===")
    print(f"Tracks: {len(merged)} | Groups: {sizes.size} | Singletons: {(sizes == 1).sum()}")
    print(f"Largest group: {sizes.max()} tracks ({sizes.max() / len(merged):.1%} of data)")
    big = merged[merged["group_id"] == sizes.idxmax()]
    print("  Largest group contents:",
          big.groupby(["language", "source"]).size().to_dict(),
          "| reasons:", big["link_reasons"].value_counts().to_dict())
    print("Tracks linked, by reason and source:")
    print(pd.crosstab(merged["link_reasons"], merged["source"]).to_string())
    mixed = merged.groupby("group_id")["source"].nunique()
    if (mixed > 1).any():
        print(f"WARNING: {(mixed > 1).sum()} groups mix AI and Human tracks - inspect near_duplicate_pairs.csv")
    if not pairs.empty:
        print(f"\nLyric pairs >= {LYRIC_JACCARD_THRESHOLD / 2:.2f} written for review: {len(pairs)} "
              f"(linked at >= {LYRIC_JACCARD_THRESHOLD}: {int(pairs['linked'].sum())})")
        print("Top pairs:")
        print(pairs.head(15).to_string(index=False))


def leakage_of_original_split(merged, edges):
    train_idx, val_idx, test_idx = reproduce_original_split(merged)
    fit_ids = set(merged.iloc[np.concatenate([train_idx, val_idx])]["song_id"])
    fit_groups = set(merged.iloc[np.concatenate([train_idx, val_idx])]["group_id"])
    test = merged.iloc[test_idx].copy()
    test["group_seen_in_train_or_val"] = test["group_id"].isin(fit_groups)
    test_ids = set(test["song_id"])
    for reason in sorted({r for _, _, r in edges}):
        leaked = {a for a, b, r in edges if r == reason and a in test_ids and b in fit_ids} | \
                 {b for a, b, r in edges if r == reason and b in test_ids and a in fit_ids}
        test[f"direct_{reason}"] = test["song_id"].isin(leaked)
    summary = (test.groupby(["language", "source"])["group_seen_in_train_or_val"]
               .agg(["mean", "sum", "size"]).rename(columns={"mean": "share_leaked",
                                                              "sum": "n_leaked", "size": "n_test"}))
    summary["share_leaked"] = (summary["share_leaked"] * 100).round(1)
    for col in [c for c in test.columns if c.startswith("direct_")]:
        summary[col.replace("direct_", "n_")] = test.groupby(["language", "source"])[col].sum()
    summary.to_csv(REVISION_DIR / "leakage_original_split.csv", encoding="utf-8-sig")
    print("\n=== Leakage in the ORIGINAL (paper) split ===")
    print(f"Test tracks: {len(test)} (paper reports 359)")
    print(f"Test tracks whose group also appears in train/val: "
          f"{int(test['group_seen_in_train_or_val'].sum())} "
          f"({test['group_seen_in_train_or_val'].mean():.1%})")
    print(summary.to_string())


EXPECTED_SCRIPT = {"English": "latin", "Hindi": "devanagari", "Chinese": "han"}


def add_script_flag(merged):
    """
    Flags tracks whose transcript is mostly in a different script from the labelled
    language (e.g. English-language songs inside the Mandarin human set).
    Uses outputs/revision/C_transcript_scripts.csv from audit_dataset.py.
    """
    path = REVISION_DIR / "C_transcript_scripts.csv"
    if not path.exists():
        print("\n(C_transcript_scripts.csv not found - run audit_dataset.py first to get the script flag)")
        merged["script_matches_language"] = True
        return merged
    scripts = pd.read_csv(path)[["song_id", "dominant_script"]].drop_duplicates("song_id")
    merged = merged.merge(scripts, on="song_id", how="left")
    merged["script_matches_language"] = [
        ds == EXPECTED_SCRIPT.get(lang) for ds, lang in zip(merged["dominant_script"], merged["language"])]
    flagged = merged[~merged["script_matches_language"]]
    print("\n=== Tracks whose transcript is not in the expected script ===")
    print(pd.crosstab([flagged["language"], flagged["source"]], flagged["dominant_script"].fillna("missing")).to_string()
          if len(flagged) else "None")
    flagged[["song_id", "source", "language", "final_genre", "title", "dominant_script"]] \
        .to_csv(REVISION_DIR / "script_mismatch_tracks.csv", index=False, encoding="utf-8-sig")
    print(f"Flagged: {len(flagged)} tracks -> script_mismatch_tracks.csv (listen to a few to confirm)")
    return merged


def make_grouped_splits(merged):
    strata = merged[["source", "language", "final_genre"]].astype(str).agg("|".join, axis=1).values
    groups = merged["group_id"].values
    out = merged[["song_id", "source", "language", "final_genre", "group_id",
                  "script_matches_language"]].copy()

    # One grouped 60/20/20 split for the headline table (fold 0 = test, fold 1 = val)
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    fold_of = np.empty(len(merged), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(merged)), strata, groups)):
        fold_of[idx] = k
    out["holdout_split"] = np.select([fold_of == 0, fold_of == 1], ["test", "val"], "train")

    # Repeated grouped CV for mean +/- std reporting
    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=r)
        col = np.empty(len(merged), dtype=int)
        for k, (_, idx) in enumerate(sgkf.split(np.zeros(len(merged)), strata, groups)):
            col[idx] = k
        out[f"rep{r}_fold"] = col

    # Safety checks
    for col in ["holdout_split"] + [f"rep{r}_fold" for r in range(N_REPEATS)]:
        crossing = out.groupby("group_id")[col].nunique()
        assert (crossing == 1).all(), f"Group crosses folds in {col}"
    out.to_csv(REVISION_DIR / "splits.csv", index=False, encoding="utf-8-sig")

    print("\n=== New grouped hold-out split ===")
    print(pd.crosstab([out["language"], out["source"]], out["holdout_split"]).to_string())
    print("\nFold sizes, repeat 0:", out["rep0_fold"].value_counts().sort_index().to_dict())
    print("No group crosses train/val/test or any CV fold: OK")


if __name__ == "__main__":
    preflight(("audio", "lyrics", "embeddings", "transcripts"))
    merged = load_merged()
    print("Modelled tracks:", len(merged))
    merged, pairs, edges = build_groups(merged)
    if not pairs.empty:
        pairs.to_csv(REVISION_DIR / "near_duplicate_pairs.csv", index=False, encoding="utf-8-sig")
    merged[["song_id", "source", "language", "final_genre", "artist", "primary_artist",
            "group_id", "link_reasons", "transcript_unique_3grams", "transcript_usable"]].to_csv(REVISION_DIR / "groups.csv", index=False, encoding="utf-8-sig")
    report_groups(merged, pairs)
    leakage_of_original_split(merged, edges)
    merged = add_script_flag(merged)
    make_grouped_splits(merged)
    print("\nSaved to", REVISION_DIR)
