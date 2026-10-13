"""
Step 3: put every track through the same processing chain.

Why: the original files came through different routes.
  Human: YouTube best audio (usually Opus/AAC) -> yt-dlp -> MP3 192 kbps (two lossy stages)
  AI:    Suno / Muse MP3 at 160 or 192 kbps (one lossy stage)
A classifier can learn those differences instead of the music. This script creates
harmonised copies so the experiments can be re-run on audio that shares one chain.

Two modes (run both; each becomes a row in your before/after table):
  common  : decode -> loudness to -14 LUFS (peak <= -1 dBFS) -> MP3 128 kbps CBR
            (128 kbps is below every file in the corpus, so this final stage dominates)
  matched : decode -> loudness to -14 LUFS -> Opus 128 kbps -> decode -> MP3 192 kbps
            (reproduces the YouTube + yt-dlp chain on every track, AI included)

Originals are never modified. Output:
  dataset_harmonised/<mode>/<song_id>.mp3
  outputs/revision/harmonised_<mode>.csv   (loudness before/after, gain, peak, errors)

Requires ffmpeg with libmp3lame and libopus on PATH (the gyan.dev "full" build on Windows has both).
Run from the project folder:
    python harmonise_audio.py --mode common
    python harmonise_audio.py --mode matched
"""
import argparse
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyloudnorm as pyln

from common import REVISION_DIR, load_audio_ok, select_per_cell

SR = 44100
TARGET_LUFS = -14.0
PEAK_CEILING_DBFS = -1.0
OUT_ROOT = Path("dataset_harmonised")


def run(cmd, input_bytes=None):
    proc = subprocess.run(cmd, input=input_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "ignore")[-400:])
    return proc.stdout


def probe_channels(path):
    out = run(["ffprobe", "-v", "error", "-select_streams", "a:0",
               "-show_entries", "stream=channels", "-of", "csv=p=0", str(path)])
    try:
        return max(1, min(2, int(out.decode().strip().splitlines()[0])))
    except Exception:
        return 2


def decode(path, channels):
    """Decode any audio file to float32 at 44.1 kHz, keeping mono files mono."""
    raw = run(["ffmpeg", "-v", "error", "-i", str(path), "-vn",
               "-ac", str(channels), "-ar", str(SR), "-f", "f32le", "-acodec", "pcm_f32le", "-"])
    return np.frombuffer(raw, dtype=np.float32).reshape(-1, channels).copy()


def encode(audio, channels, out_path, codec_args):
    run(["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", str(channels),
         "-i", "-", *codec_args, str(out_path)], input_bytes=audio.astype(np.float32).tobytes())


def loudness(audio, meter):
    try:
        value = meter.integrated_loudness(audio if audio.shape[1] > 1 else audio[:, 0])
        return float(value) if np.isfinite(value) else np.nan
    except Exception:
        return np.nan


def harmonise_one(song_id, in_path, mode):
    out_path = OUT_ROOT / mode / f"{song_id}.mp3"
    rec = {"song_id": song_id, "original_path": in_path, "audio_path": str(out_path), "ok": False, "error": ""}
    try:
        channels = probe_channels(in_path)
        audio = decode(in_path, channels)
        meter = pyln.Meter(SR)
        lufs_in = loudness(audio, meter)
        peak_in = float(np.max(np.abs(audio))) if audio.size else 0.0

        # Gain to target loudness, but never let the sample peak exceed the ceiling.
        # No limiter is used, so no new distortion is introduced; quiet peaky tracks
        # may therefore end slightly below -14 LUFS (recorded in lufs_out).
        gain_db = (TARGET_LUFS - lufs_in) if np.isfinite(lufs_in) else 0.0
        ceiling = 10 ** (PEAK_CEILING_DBFS / 20)
        if peak_in * 10 ** (gain_db / 20) > ceiling:
            gain_db = 20 * np.log10(ceiling / (peak_in + 1e-12))
            rec["peak_limited_gain"] = True
        else:
            rec["peak_limited_gain"] = False
        audio = audio * (10 ** (gain_db / 20))

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if mode == "common":
            encode(audio, channels, out_path, ["-c:a", "libmp3lame", "-b:a", "128k"])
        elif mode == "matched":
            with tempfile.TemporaryDirectory() as tmp:
                opus_path = Path(tmp) / "stage1.opus"
                encode(audio, channels, opus_path, ["-c:a", "libopus", "-b:a", "128k", "-vbr", "on", "-ar", "48000"])
                stage1 = decode(opus_path, channels)
                encode(stage1, channels, out_path, ["-c:a", "libmp3lame", "-b:a", "192k"])
        else:
            raise ValueError(mode)

        check = decode(out_path, channels)
        rec.update({
            "channels": channels,
            "lufs_in": round(lufs_in, 2) if np.isfinite(lufs_in) else None,
            "gain_db": round(float(gain_db), 2),
            "lufs_out": round(loudness(check, meter), 2),
            "peak_out_dbfs": round(20 * np.log10(np.max(np.abs(check)) + 1e-12), 2),
            "duration_s": round(len(check) / SR, 2),
            "ok": True,
        })
    except Exception as exc:
        rec["error"] = str(exc)[:400]
    return rec


def check_ffmpeg(mode):
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"]).decode("utf-8", "ignore")
    needed = ["libmp3lame"] + (["libopus"] if mode == "matched" else [])
    missing = [e for e in needed if e not in encoders]
    if missing:
        raise SystemExit(f"ffmpeg is missing encoders: {missing}. Install a full ffmpeg build.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["common", "matched"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--limit", type=int, default=0, help="process only N tracks (for a quick test)")
    parser.add_argument("--per-cell", type=int, default=0,
                        help="balanced random subset: N tracks per language x source cell (e.g. 60 = 360 tracks)")
    args = parser.parse_args()

    check_ffmpeg(args.mode)
    audio = load_audio_ok()
    if args.per_cell:
        audio = select_per_cell(audio, args.per_cell).reset_index(drop=True)
        print(f"Balanced subset: {len(audio)} tracks ({args.per_cell} per language x source cell)")
    todo = audio[["song_id", "audio_path"]].copy()
    out_csv = REVISION_DIR / f"harmonised_{args.mode}.csv"

    done = pd.DataFrame()
    if out_csv.exists():
        done = pd.read_csv(out_csv)
        ok_ids = set(done.loc[done["ok"].astype(str).str.lower() == "true", "song_id"].astype(str))
        todo = todo[~todo["song_id"].isin(ok_ids)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Mode: {args.mode} | to process: {len(todo)} | workers: {args.workers}")

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(harmonise_one, r.song_id, r.audio_path, args.mode)
                   for r in todo.itertuples(index=False)]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            if not rec["ok"]:
                print(f"  ERROR {rec['song_id']}: {rec['error'][:150]}")
            if n % 50 == 0 or n == len(futures):
                print(f"  {n}/{len(futures)} done")
                pd.concat([done, pd.DataFrame(records)]).drop_duplicates("song_id", keep="last") \
                  .to_csv(out_csv, index=False, encoding="utf-8-sig")

    result = pd.concat([done, pd.DataFrame(records)]).drop_duplicates("song_id", keep="last")
    result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    ok = result["ok"].astype(str).str.lower() == "true"
    merged = result[ok].merge(audio[["song_id", "source"]], on="song_id")
    print(f"\nFinished: {ok.sum()} OK, {(~ok).sum()} errors -> {out_csv}")
    if len(merged):
        print("Loudness after harmonisation (mean LUFS by source):")
        print(merged.groupby("source")[["lufs_in", "lufs_out"]].mean().round(2).to_string())
        print("Tracks whose gain was reduced to protect peaks:",
              int(merged["peak_limited_gain"].astype(str).str.lower().eq("true").sum()))


if __name__ == "__main__":
    main()
