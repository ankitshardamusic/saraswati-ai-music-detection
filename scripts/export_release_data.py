"""
Export the shareable part of the Saraswati data into data/release/.

Shared (numbers and identifiers only):
  tracks.csv               song_id, source, language, genre, title, artist, public source link/ID,
                           grouped split, group id, transcript-script flag
  audio_features.csv       all audio features (78 used for modelling + the excluded ones)
  lyrics_features.csv      handcrafted lyric statistics (numbers only)
  lyrics_embeddings.npy    384-d multilingual sentence embeddings (+ lyrics_embedding_ids.csv)
  muq_eval_scores.csv      per-track MuQ-Eval scores
  results/                 result tables from outputs/fusion_results and outputs/revision

NOT shared: audio files, Whisper transcripts or any lyric text, separated vocals,
local file paths. (Commercial recordings and lyrics are copyrighted.)

Run from the project folder (the one containing outputs/):
    python scripts/export_release_data.py
"""
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("outputs")
DATA = Path("dataset_collection_clean")
REL = Path("data") / "release"

TEXT_LIKE = re.compile(r"(text|lyric_line|transcript|vocals_path|audio_path|_path$|error|query|"
                       r"matched_title|matched_uploader|matched_channel|search_error|filename)", re.I)
LINK_COLS = ["youtube_video_id", "youtube_url", "bilibili_video_id", "bilibili_url",
             "media_platform", "media_id", "media_url", "source_dataset", "source_track_id", "model"]


def ok_rows(df, col):
    if col not in df.columns:
        return df
    return df[df[col].astype(str).str.strip().str.lower().isin(["true", "1", "yes"])]


def numeric_only(df, keep=("song_id",)):
    cols = list(keep)
    for c in df.columns:
        if c in keep or TEXT_LIKE.search(c):
            continue
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return df[cols]


def read_manifest_links():
    frames = []
    for csv in list((DATA / "manifests").glob("*_final_with_audio.csv")) + list((DATA / "ai").rglob("*.csv")):
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        if "song_id" not in df.columns and "new_song_id" in df.columns:
            df["song_id"] = df["new_song_id"]
        if "song_id" not in df.columns:
            continue
        cols = ["song_id"] + [c for c in LINK_COLS if c in df.columns]
        frames.append(df[cols].astype({"song_id": str}))
    if not frames:
        return pd.DataFrame(columns=["song_id"])
    links = pd.concat(frames, ignore_index=True)
    links["song_id"] = links["song_id"].str.strip()
    return links.drop_duplicates("song_id", keep="last")


def main():
    if not (OUT / "audio_features.csv").exists():
        sys.exit("Run this from the project folder that contains outputs/audio_features.csv")
    REL.mkdir(parents=True, exist_ok=True)

    audio = ok_rows(pd.read_csv(OUT / "audio_features.csv", low_memory=False), "ok")
    audio["song_id"] = audio["song_id"].astype(str).str.strip()
    audio = audio.drop_duplicates("song_id", keep="last")

    tracks = audio[[c for c in ["song_id", "source", "language", "final_genre", "title", "artist",
                                "duration_seconds"] if c in audio.columns]].copy()
    tracks = tracks.merge(read_manifest_links(), on="song_id", how="left")
    REV = OUT / "revision" if (OUT / "revision").exists() else OUT / "week1"
    splits = REV / "splits.csv"
    if splits.exists():
        s = pd.read_csv(splits)
        keep = [c for c in ["song_id", "group_id", "holdout_split", "script_matches_language"] if c in s.columns]
        tracks = tracks.merge(s[keep], on="song_id", how="left")
    tracks.to_csv(REL / "tracks.csv", index=False, encoding="utf-8-sig")

    numeric_only(audio).to_csv(REL / "audio_features.csv", index=False, encoding="utf-8-sig")

    lyr_path = OUT / "lyrics_features.csv"
    if lyr_path.exists():
        lyr = pd.read_csv(lyr_path, low_memory=False)
        lyr["song_id"] = lyr["song_id"].astype(str).str.strip()
        lyr = lyr.drop_duplicates("song_id", keep="last")
        numeric_only(lyr).to_csv(REL / "lyrics_features.csv", index=False, encoding="utf-8-sig")

    emb_path, emb_meta = OUT / "lyrics_embeddings.npy", OUT / "lyrics_embedding_metadata.csv"
    if emb_path.exists() and emb_meta.exists():
        np.save(REL / "lyrics_embeddings.npy", np.load(emb_path, allow_pickle=False))
        pd.read_csv(emb_meta)[["song_id"]].to_csv(REL / "lyrics_embedding_ids.csv", index=False)

    muq = OUT / "muq_eval_scores.csv"
    if muq.exists():
        m = pd.read_csv(muq, low_memory=False)
        keep = ["song_id"] + [c for c in m.columns if c.startswith("muq_") or c in ("n_windows", "duration_sec")]
        m[keep].to_csv(REL / "muq_eval_scores.csv", index=False, encoding="utf-8-sig")

    res = REL / "results"
    res.mkdir(exist_ok=True)
    for folder in [OUT / "fusion_results", OUT / "revision", OUT / "week1"]:
        for csv in folder.glob("*.csv") if folder.exists() else []:
            if csv.name == "technical_issue_report.csv":   # per-track issue labels for named songs; not needed
                continue
            df = pd.read_csv(csv, low_memory=False)
            df = df[[c for c in df.columns if not TEXT_LIKE.search(c)]]
            df.to_csv(res / csv.name, index=False, encoding="utf-8-sig")

    # Final safety check: no lyric text and no local paths in anything exported
    for f in REL.rglob("*.csv"):
        txt = f.read_text(encoding="utf-8-sig", errors="ignore")
        if re.search(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?!/)|dataset_collection_clean|outputs[\\/]transcripts", txt):
            sys.exit(f"STOP: {f} still contains a local path - please send me this message")
    total = sum(p.stat().st_size for p in REL.rglob("*") if p.is_file())
    print(f"Exported {len(tracks)} tracks to {REL} ({total / 1e6:.1f} MB)")
    for p in sorted(REL.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(REL)}  ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
