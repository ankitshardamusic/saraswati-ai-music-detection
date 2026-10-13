"""
Shared settings and loaders for the revision scripts.

Run every script from the same project folder where you ran the notebooks,
so that `outputs/` and `dataset_collection_clean/` resolve correctly.
"""
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

# Paths (same as in the notebooks)
OUTPUT_DIR = Path("outputs")
AUDIO_CSV = OUTPUT_DIR / "audio_features.csv"
LYRICS_CSV = OUTPUT_DIR / "lyrics_features.csv"
LYRICS_EMB_NPY = OUTPUT_DIR / "lyrics_embeddings.npy"
LYRICS_EMB_META = OUTPUT_DIR / "lyrics_embedding_metadata.csv"
TRANSCRIPTS_DIR = OUTPUT_DIR / "transcripts"
DATA_DIR = Path("dataset_collection_clean")

# results of these scripts go to outputs/revision (an older outputs/week1 folder is still used if it exists)
REVISION_DIR = OUTPUT_DIR / "revision"
if not REVISION_DIR.exists() and (OUTPUT_DIR / "week1").exists():
    REVISION_DIR = OUTPUT_DIR / "week1"
REVISION_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
LANGUAGES = ["English", "Hindi", "Chinese"]

# Windows consoles can crash when printing Hindi/Chinese text; print safely instead
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def preflight(required=("audio", "lyrics", "embeddings")):
    """Stop early with a clear message if the script is run from the wrong folder."""
    files = {
        "audio": [AUDIO_CSV],
        "lyrics": [LYRICS_CSV],
        "embeddings": [LYRICS_EMB_NPY, LYRICS_EMB_META],
        "transcripts": [TRANSCRIPTS_DIR],
    }
    print("Working folder:", Path.cwd())
    missing = [str(p) for key in required for p in files[key] if not p.exists()]
    for key in files:
        for p in files[key]:
            print(f"  {'found  ' if p.exists() else 'MISSING'}  {p}")
    if missing:
        print("\nSTOP: required files are missing. Run this script from the folder that "
              "contains the 'outputs' folder (the same folder as your notebooks).")
        sys.exit(1)


def true_mask(series):
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes"])


def load_audio_ok():
    """Audio rows exactly as the fusion notebook filters them."""
    df = pd.read_csv(AUDIO_CSV, low_memory=False)
    df["song_id"] = df["song_id"].astype(str).str.strip()
    if "ok" in df.columns:
        df = df.loc[true_mask(df["ok"])].copy()
    df = df.drop_duplicates("song_id", keep="last")
    return df[df["source"].isin(["AI", "Human"])].reset_index(drop=True)


def load_lyrics_ok(all_rows=False):
    df = pd.read_csv(LYRICS_CSV, low_memory=False)
    df["song_id"] = df["song_id"].astype(str).str.strip()
    if not all_rows and "transcript_ok" in df.columns:
        df = df.loc[true_mask(df["transcript_ok"])].copy()
    return df.drop_duplicates("song_id", keep="last").reset_index(drop=True)


def load_embeddings():
    matrix = np.load(LYRICS_EMB_NPY, allow_pickle=False)
    meta = pd.read_csv(LYRICS_EMB_META, low_memory=False)
    if len(matrix) != len(meta):
        raise ValueError(f"Embedding mismatch: {len(matrix)} vs {len(meta)}")
    ids = meta["song_id"].astype(str).str.strip().values
    emb = pd.DataFrame(matrix, columns=[f"emb_{i:03d}" for i in range(matrix.shape[1])])
    emb.insert(0, "song_id", ids)
    return emb.drop_duplicates("song_id", keep="last").reset_index(drop=True)


def load_merged():
    """
    Rebuilds the `merged` table in the same row order as the fusion notebook,
    so the original train/val/test split can be reproduced exactly.
    Only the columns needed for splitting are kept.
    """
    audio = load_audio_ok()
    lyrics = load_lyrics_ok()
    emb = load_embeddings()
    meta_cols = [c for c in ["song_id", "title", "artist", "source", "language",
                             "final_genre", "audio_path"] if c in audio.columns]
    merged = (
        audio[meta_cols]
        .merge(lyrics[["song_id"]], on="song_id", how="inner")
        .merge(emb[["song_id"]], on="song_id", how="inner")
        .drop_duplicates("song_id", keep="last")
        .reset_index(drop=True)
    )
    return merged


def reproduce_original_split(merged):
    """Exact copy of the split in 03_multimodal_fusion_final.ipynb (cell 14)."""
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(merged))
    strata = merged[["source", "language", "final_genre"]].astype(str).agg("|".join, axis=1)
    train_val_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=strata, random_state=RANDOM_SEED)
    val_fraction = 0.20 / (0.60 + 0.20)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_fraction,
        stratify=strata.iloc[train_val_idx], random_state=RANDOM_SEED)
    return train_idx, val_idx, test_idx


# Artist names: split multi-artist credits and take the primary artist
_ARTIST_SPLIT = re.compile(
    r"\s*(?:,|;|/|&|\+|、|，|和|\bfeat\.?\b|\bft\.?\b|\bfeaturing\b|\bwith\b|\bx\b|\band\b)\s*",
    flags=re.IGNORECASE,
)


def split_artists(name):
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return []
    text = unicodedata.normalize("NFKC", str(name)).casefold().strip()
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", text)  # drop bracketed notes
    parts = [p.strip(" .-_'\"") for p in _ARTIST_SPLIT.split(text)]
    parts = [re.sub(r"\s+", " ", p) for p in parts if p and p not in {"nan", "none", "unknown", "various artists", "various", "va"}]
    return parts


def primary_artist(name):
    parts = split_artists(name)
    return parts[0] if parts else ""


def select_per_cell(df, n, seed=42):
    """Balanced random subset: up to n tracks per language x source cell (same tracks every run)."""
    parts = [g.sample(min(len(g), n), random_state=seed) for _, g in df.groupby(["language", "source"])]
    return pd.concat(parts).sort_index()
