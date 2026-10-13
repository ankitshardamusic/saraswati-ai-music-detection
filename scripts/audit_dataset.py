"""
Step 1: dataset audit (no audio is modified).

Answers three questions:
  A. Which tracks drop out between 1,800 audio files and 1,791 modelled tracks, and why?
  B. Can the file format alone (codec, bitrate, sample rate, channels) predict AI vs Human?
     If yes, that is direct evidence of an acquisition confound.
  C. What script are the Hindi / Mandarin transcripts actually written in, and how does
     that relate to Whisper's detected language (e.g. Hindi detected as Urdu)?

Outputs go to outputs/revision/. Run from the project folder:
    python audit_dataset.py
"""
import json
import shutil
import subprocess
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from common import (TRANSCRIPTS_DIR, REVISION_DIR, RANDOM_SEED, load_audio_ok,
                          load_lyrics_ok, load_embeddings, load_merged, preflight)


# ---------------------------------------------------------------- A. dropped tracks
def audit_dropped_tracks():
    audio = load_audio_ok()
    lyrics_all = load_lyrics_ok(all_rows=True)
    lyrics_ok_ids = set(load_lyrics_ok()["song_id"])
    emb_ids = set(load_embeddings()["song_id"])
    merged_ids = set(load_merged()["song_id"])

    dropped = audio[~audio["song_id"].isin(merged_ids)].copy()
    reasons = []
    lyr = lyrics_all.set_index("song_id")
    for sid in dropped["song_id"]:
        if sid not in lyr.index:
            reasons.append("not in lyrics_features.csv")
        elif sid not in lyrics_ok_ids:
            err = str(lyr.at[sid, "transcript_error"]) if "transcript_error" in lyr.columns else ""
            reasons.append(f"transcript_ok=False ({err[:120]})")
        elif sid not in emb_ids:
            reasons.append("no lyric embedding")
        else:
            reasons.append("other")
    dropped["drop_reason"] = reasons
    cols = [c for c in ["song_id", "source", "language", "final_genre", "title", "drop_reason"]
            if c in dropped.columns]
    dropped[cols].to_csv(REVISION_DIR / "A_dropped_tracks.csv", index=False, encoding="utf-8-sig")

    print("\n=== A. Tracks with audio features but excluded from modelling ===")
    print(f"Audio OK: {len(audio)} | Modelled (merged): {len(merged_ids)} | Dropped: {len(dropped)}")
    if len(dropped):
        print(dropped[cols].to_string(index=False))


# ---------------------------------------------------------------- B. file-format confound
def ffprobe(path):
    cmd = ["ffprobe", "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate:format=bit_rate,duration",
           "-of", "json", str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        data = json.loads(out.stdout or "{}")
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format", {})
        bitrate = stream.get("bit_rate") or fmt.get("bit_rate")
        return {
            "codec": stream.get("codec_name"),
            "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else np.nan,
            "channels": int(stream["channels"]) if stream.get("channels") else np.nan,
            "bitrate_kbps": round(int(bitrate) / 1000) if bitrate not in (None, "N/A") else np.nan,
            "duration_s": float(fmt["duration"]) if fmt.get("duration") else np.nan,
            "probe_ok": True,
        }
    except Exception as exc:
        return {"probe_ok": False, "probe_error": str(exc)[:200]}


def mutagen_probe(path):
    """Fallback when ffprobe is not installed (mutagen was used in your data notebook)."""
    try:
        from mutagen import File as MutagenFile
        info = MutagenFile(str(path)).info
        return {
            "codec": type(info).__name__.replace("Info", "").lower(),
            "sample_rate": getattr(info, "sample_rate", np.nan),
            "channels": getattr(info, "channels", np.nan),
            "bitrate_kbps": round(getattr(info, "bitrate", 0) / 1000) or np.nan,
            "duration_s": getattr(info, "length", np.nan),
            "probe_ok": True,
        }
    except Exception as exc:
        return {"probe_ok": False, "probe_error": str(exc)[:200]}


def audit_file_format():
    audio = load_audio_ok()
    paths = audio["audio_path"].astype(str).tolist()
    missing_files = sum(not Path(p).is_file() for p in paths)
    if missing_files:
        print(f"WARNING: {missing_files} audio paths in audio_features.csv do not exist from this folder.")
    probe = ffprobe if shutil.which("ffprobe") else mutagen_probe
    print("Reading file format with:", "ffprobe" if probe is ffprobe else "mutagen (ffprobe not found)")
    with ThreadPoolExecutor(max_workers=8) as pool:
        probes = list(pool.map(probe, paths))
    probes = pd.DataFrame(probes)
    for col in ["codec", "sample_rate", "channels", "bitrate_kbps", "duration_s", "probe_error"]:
        if col not in probes.columns:
            probes[col] = np.nan
    meta = pd.concat([audio[["song_id", "source", "language", "final_genre", "audio_path"]],
                      probes], axis=1)
    meta.to_csv(REVISION_DIR / "B_file_metadata.csv", index=False, encoding="utf-8-sig")

    print("\n=== B. File format by source and language ===")
    n_fail = int((~meta["probe_ok"].astype(bool)).sum())
    print("Probe failures:", n_fail)
    if n_fail == len(meta):
        print("Every file failed. First error:", meta["probe_error"].dropna().head(1).tolist())
        return
    for col in ["codec", "bitrate_kbps", "sample_rate", "channels"]:
        table = pd.crosstab([meta["source"], meta["language"]], meta[col].astype(str))
        print(f"\n-- {col} --")
        print(table.to_string())
        table.to_csv(REVISION_DIR / f"B_crosstab_{col}.csv", encoding="utf-8-sig")

    # Can format metadata ALONE predict the label? (0.5 = no leak, 1.0 = total leak)
    ok = meta["probe_ok"].astype(bool)
    X = meta.loc[ok, ["codec", "bitrate_kbps", "sample_rate", "channels"]].astype(str)
    y = (meta.loc[ok, "source"] == "AI").astype(int)
    model = make_pipeline(OneHotEncoder(handle_unknown="ignore"), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_SEED)
    auc = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    print(f"\nAUROC for predicting AI vs Human from file format ONLY: "
          f"{auc.mean():.3f} +/- {auc.std():.3f}")
    print("  (Report this in the paper. Anything clearly above 0.5 is an acquisition confound.)")
    for lang in sorted(meta["language"].unique()):
        m = ok & (meta["language"] == lang)
        if meta.loc[m, "source"].nunique() == 2:
            a = cross_val_score(model, meta.loc[m, ["codec", "bitrate_kbps", "sample_rate", "channels"]].astype(str),
                                (meta.loc[m, "source"] == "AI").astype(int), cv=cv, scoring="roc_auc")
            print(f"  {lang:<8} AUROC = {a.mean():.3f}")


# ---------------------------------------------------------------- C. transcript scripts
SCRIPT_RANGES = {
    "devanagari": [(0x0900, 0x097F), (0xA8E0, 0xA8FF)],
    "arabic_urdu": [(0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)],
    "latin": [(0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F)],
    "han": [(0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0xF900, 0xFAFF)],
    "gurmukhi": [(0x0A00, 0x0A7F)],
    "kana_hangul": [(0x3040, 0x30FF), (0xAC00, 0xD7AF)],
}


def script_of(ch):
    cp = ord(ch)
    for name, ranges in SCRIPT_RANGES.items():
        if any(lo <= cp <= hi for lo, hi in ranges):
            return name
    return "other"


def script_profile(text):
    letters = [c for c in text if c.isalpha()]
    counts = {name: 0 for name in list(SCRIPT_RANGES) + ["other"]}
    for c in letters:
        counts[script_of(c)] += 1
    total = max(1, len(letters))
    profile = {f"frac_{k}": round(v / total, 4) for k, v in counts.items()}
    profile["n_letters"] = len(letters)
    profile["dominant_script"] = max(counts, key=counts.get) if letters else "empty"
    return profile


def audit_transcripts():
    lyrics = load_lyrics_ok(all_rows=True)
    rows = []
    for _, r in lyrics.iterrows():
        path = TRANSCRIPTS_DIR / f"{r['song_id']}.txt"
        text = path.read_text(encoding="utf-8", errors="ignore") if path.exists() else ""
        rec = {"song_id": r["song_id"], "language": r.get("language"), "source": r.get("source"),
               "whisper_detected_lang": r.get("whisper_detected_lang"),
               "transcript_found": path.exists()}
        rec.update(script_profile(text))
        rows.append(rec)
    prof = pd.DataFrame(rows)
    prof.to_csv(REVISION_DIR / "C_transcript_scripts.csv", index=False, encoding="utf-8-sig")

    print("\n=== C. Transcript script audit ===")
    print("Transcripts missing on disk:", int((~prof["transcript_found"]).sum()))
    for lang in ["Hindi", "Chinese", "English"]:
        part = prof[prof["language"] == lang]
        if part.empty:
            continue
        print(f"\n-- {lang}: dominant script of transcript --")
        print(pd.crosstab(part["dominant_script"], part["source"]).to_string())
        print(f"\n-- {lang}: Whisper detected language vs dominant script --")
        print(pd.crosstab(part["whisper_detected_lang"].fillna("none"),
                          part["dominant_script"]).to_string())

    hindi = prof[prof["language"] == "Hindi"]
    if not hindi.empty:
        dev = (hindi["dominant_script"] == "devanagari").groupby(hindi["source"]).mean().mul(100).round(1)
        hi_ur = hindi["whisper_detected_lang"].astype(str).isin(["hi", "ur"]).groupby(hindi["source"]).mean().mul(100).round(1)
        print("\nHindi transcripts written mainly in Devanagari (%):", dev.to_dict())
        print("Hindi tracks detected as 'hi' OR 'ur' (%):          ", hi_ur.to_dict())
        print("  If Devanagari is high, the transcripts are usable and the paper's 39.7% / 20.7%")
        print("  'language match' figures reflect Whisper's Hindi/Urdu language-ID confusion,")
        print("  not wrong-language transcripts. Rewrite that sentence accordingly.")

    chinese = prof[prof["language"] == "Chinese"]
    if not chinese.empty:
        print("\nMandarin set - Whisper detected languages (check for 'yue' = Cantonese):")
        print(pd.crosstab(chinese["whisper_detected_lang"].fillna("none"), chinese["source"]).to_string())


if __name__ == "__main__":
    preflight(("audio", "lyrics", "embeddings"))
    for step in (audit_dropped_tracks, audit_file_format, audit_transcripts):
        try:
            step()
        except Exception:
            print(f"\n!!! {step.__name__} failed - please send me this error:")
            traceback.print_exc()
    print("\nFinished. Tables saved to", REVISION_DIR)
