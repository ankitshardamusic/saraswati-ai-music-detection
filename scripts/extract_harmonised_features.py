"""
Step 4: re-extract the audio features from the harmonised files.

The feature functions below are copied from 01_audio_features_final.ipynb with the
logic unchanged. Only indentation errors were fixed: in the uploaded notebook,
several lines (e.g. `crest = ...`, `stft = ...`, `f0,voiced_flag,_ = ...`, `data,_=sf.read(...)`)
start at column 0 inside functions, so those cells do not run as saved. Fix them in the
notebook too before you release the code.

Runs in parallel (pyin is the slow part), so it takes hours instead of 45-60 hours.

Output:
  outputs/audio_features_<mode>.csv          same columns as outputs/audio_features.csv
  outputs/revision/feature_shift_<mode>.csv     per feature: how much harmonisation changed it

Run from the project folder, after harmonise_audio.py:
    python extract_harmonised_features.py --mode common
    python extract_harmonised_features.py --mode matched
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per worker process
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from common import AUDIO_CSV, OUTPUT_DIR, REVISION_DIR, select_per_cell, true_mask

SAMPLE_RATE = 22050
N_MFCC = 13
N_CHROMA = 12


# ------------------------------------------------------------------ copied feature code
def extract_loudness(y, sr):
    import librosa, pyloudnorm as pyln
    y = np.asarray(y, dtype=np.float32).flatten()
    meter = pyln.Meter(sr)
    try:
        lufs = float(meter.integrated_loudness(y))
    except Exception:
        lufs = np.nan
    peak = float(np.max(np.abs(y))) if len(y) else 0.0
    rms = float(np.sqrt(np.mean(y ** 2))) if len(y) else 0.0
    crest = peak / (rms + 1e-9)
    frame_rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    frame_db = librosa.amplitude_to_db(frame_rms, ref=1.0, top_db=None)
    clips = int(np.sum(np.abs(y) >= 1.0))
    silence_ratio = float(np.mean(frame_db < -60)) if len(frame_db) else np.nan
    dynamic_range = (float(np.percentile(frame_db, 95) - np.percentile(frame_db, 5))
                     if len(frame_db) else np.nan)
    return {
        'lufs_integrated': round(lufs, 3) if np.isfinite(lufs) else None,
        'sample_peak_dbfs': round(20 * np.log10(peak + 1e-9), 3),
        'rms_db': round(20 * np.log10(rms + 1e-9), 3),
        'crest_factor_db': round(20 * np.log10(crest + 1e-9), 3),
        'dynamic_range_db': round(dynamic_range, 3) if np.isfinite(dynamic_range) else None,
        'hard_clip_count': clips,
        'is_clipped': clips > 0,
        'silence_ratio': round(silence_ratio, 4) if np.isfinite(silence_ratio) else None,
    }


def extract_spectral(y, sr, n_mfcc=13, n_chroma=12):
    import librosa
    stft = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)
    power = np.mean(stft ** 2, axis=1)
    total_p = np.sum(power) + 1e-9
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    frame_norm = np.linalg.norm(stft, axis=0, keepdims=True)
    stft_norm = stft / (frame_norm + 1e-9)
    flux = (np.mean(np.diff(stft_norm, axis=1) ** 2, axis=0)
            if stft_norm.shape[1] > 1 else np.array([0.0]))
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=6)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta_mfccs = librosa.feature.delta(mfccs)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=n_chroma)
    out = {
        'spectral_centroid_mean': round(float(np.mean(centroid)), 3),
        'spectral_centroid_std': round(float(np.std(centroid)), 3),
        'spectral_rolloff_mean': round(float(np.mean(rolloff)), 3),
        'spectral_flatness_mean': round(float(np.mean(flatness)), 6),
        'spectral_bandwidth_mean': round(float(np.mean(bandwidth)), 3),
        'spectral_flux_mean': round(float(np.mean(flux)), 6),
        'spectral_contrast_mean': round(float(np.mean(contrast)), 4),
        'band_energy_low': round(float(np.sum(power[freqs < 250]) / total_p), 4),
        'band_energy_mid': round(float(np.sum(power[(freqs >= 250) & (freqs < 4000)]) / total_p), 4),
        'band_energy_high': round(float(np.sum(power[freqs >= 4000]) / total_p), 4),
        'chroma_mean': round(float(np.mean(chroma)), 4),
        'chroma_std': round(float(np.std(chroma)), 4),
    }
    for i in range(n_mfcc):
        out[f'mfcc_{i+1}_mean'] = round(float(np.mean(mfccs[i])), 4)
        out[f'mfcc_{i+1}_std'] = round(float(np.std(mfccs[i])), 4)
        out[f'delta_mfcc_{i+1}_mean'] = round(float(np.mean(delta_mfccs[i])), 4)
        out[f'delta_mfcc_{i+1}_std'] = round(float(np.std(delta_mfccs[i])), 4)
    return out


def extract_temporal(y, sr):
    import librosa
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(y) == 0:
        return {k: None for k in ['bpm', 'beat_regularity_std', 'beat_regularity_cv',
                                  'onset_strength_mean', 'onset_strength_std', 'zcr_mean', 'zcr_std']}
    y_norm = librosa.util.normalize(y)
    tempo, beats = librosa.beat.beat_track(y=y_norm, sr=sr)
    beat_times = librosa.frames_to_time(beats, sr=sr)
    if len(beat_times) > 2:
        intervals = np.diff(beat_times)
        ibi_std = float(np.std(intervals))
        ibi_cv = ibi_std / (float(np.mean(intervals)) + 1e-9)
    else:
        ibi_std = None
        ibi_cv = None
    onset_env = librosa.onset.onset_strength(y=y_norm, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y_norm)[0]
    tempo_value = float(np.asarray(tempo).squeeze())
    return {
        'bpm': round(tempo_value, 2),
        'beat_regularity_std': round(ibi_std, 4) if ibi_std is not None else None,
        'beat_regularity_cv': round(ibi_cv, 4) if ibi_cv is not None else None,
        'onset_strength_mean': round(float(np.mean(onset_env)), 4),
        'onset_strength_std': round(float(np.std(onset_env)), 4),
        'zcr_mean': round(float(np.mean(zcr)), 6),
        'zcr_std': round(float(np.std(zcr)), 6),
    }


def extract_pitch(y, sr):
    import librosa
    try:
        f0, voiced_flag, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                          fmax=librosa.note_to_hz('C7'), sr=sr)
        valid = voiced_flag & np.isfinite(f0)
        voiced_f0 = f0[valid]
        if len(voiced_f0) == 0:
            raise ValueError('No voiced frames')
        tuning = float(librosa.estimate_tuning(y=y, sr=sr))
        reference_a4 = 440.0 * (2.0 ** (tuning / 12.0))
        note_position = 12.0 * np.log2(voiced_f0 / reference_a4)
        nearest_note = np.round(note_position)
        nearest_frequency = reference_a4 * (2.0 ** (nearest_note / 12.0))
        deviations_cents = 1200.0 * np.log2(voiced_f0 / nearest_frequency)
        abs_dev = np.abs(deviations_cents)
        return {
            'f0_mean': round(float(np.mean(voiced_f0)), 3),
            'f0_std': round(float(np.std(voiced_f0)), 3),
            'f0_voiced_ratio': round(float(np.mean(valid)), 4),
            'tuning_offset_cents': round(tuning * 100.0, 2),
            'pitch_dev_mean_cents': round(float(np.mean(abs_dev)), 3),
            'pitch_dev_std_cents': round(float(np.std(abs_dev)), 3),
            'pitch_microtonal_ratio': round(float(np.mean(abs_dev > 20)), 4),
        }
    except Exception:
        return {k: None for k in ['f0_mean', 'f0_std', 'f0_voiced_ratio', 'tuning_offset_cents',
                                  'pitch_dev_mean_cents', 'pitch_dev_std_cents', 'pitch_microtonal_ratio']}


def extract_stereo(audio_path):
    import soundfile as sf
    try:
        data, _ = sf.read(str(audio_path), always_2d=True, dtype='float32')
        if data.shape[1] < 2:
            return {'stereo_lr_ratio': None, 'stereo_width': None, 'is_mono': True}
        left, right = data[:, 0], data[:, 1]
        rms_l = float(np.sqrt(np.mean(left ** 2)))
        rms_r = float(np.sqrt(np.mean(right ** 2)))
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        mid_rms = float(np.sqrt(np.mean(mid ** 2)))
        side_rms = float(np.sqrt(np.mean(side ** 2)))
        return {'stereo_lr_ratio': round(rms_l / (rms_r + 1e-9), 4),
                'stereo_width': round(side_rms / (mid_rms + 1e-9), 4), 'is_mono': False}
    except Exception:
        return {'stereo_lr_ratio': None, 'stereo_width': None, 'is_mono': None}


def extract_mel_stats(audio_path):
    """Same as the 'mel statistics patch' cell: first 30 s, 24 mel bands up to 8 kHz."""
    import librosa
    import scipy.stats as sp_stats
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True, duration=30)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=24, fmax=min(8000, sr / 2))
        mel_db = librosa.power_to_db(mel + 1e-12, ref=np.max)
        kurt = sp_stats.kurtosis(mel_db, axis=1, fisher=True, nan_policy='omit')
        skew = sp_stats.skew(mel_db, axis=1, nan_policy='omit')
        return {'mel_kurtosis_mean': round(float(np.nanmean(kurt)), 3),
                'mel_kurtosis_std': round(float(np.nanstd(kurt)), 3),
                'mel_skewness_std': round(float(np.nanstd(skew)), 3)}
    except Exception:
        return {'mel_kurtosis_mean': np.nan, 'mel_kurtosis_std': np.nan, 'mel_skewness_std': np.nan}


def extract_all(song_id, audio_path):
    import librosa
    warnings.filterwarnings('ignore')
    rec = {'song_id': song_id, 'audio_path': str(audio_path), 'ok': False, 'error': ''}
    try:
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        rec.update(extract_loudness(y, sr))
        rec.update(extract_spectral(y, sr, N_MFCC, N_CHROMA))
        rec.update(extract_temporal(y, sr))
        rec.update(extract_pitch(y, sr))
        rec.update(extract_stereo(audio_path))
        rec.update(extract_mel_stats(audio_path))
        rec['duration_seconds'] = round(float(librosa.get_duration(y=y, sr=sr)), 2)
        rec['ok'] = True
    except Exception as exc:
        rec['error'] = str(exc)[:300]
    return rec


# ------------------------------------------------------------------ runner
def compare_with_original(new_df, mode):
    orig = pd.read_csv(AUDIO_CSV, low_memory=False)
    orig = orig.loc[true_mask(orig['ok'])].drop_duplicates('song_id', keep='last')
    orig['song_id'] = orig['song_id'].astype(str).str.strip()
    skip = {'song_id', 'title', 'artist', 'language', 'final_genre', 'source', 'audio_path',
            'original_audio_path', 'ok', 'error', 'is_clipped', 'is_mono'}
    only_orig = sorted(set(orig.columns) - set(new_df.columns) - skip)
    if only_orig:
        print("\nColumns in the ORIGINAL csv that this script does not produce "
              "(they come from older notebook versions; check none of them are in your 78 features):")
        print("  " + ", ".join(only_orig))

    both = orig.merge(new_df, on='song_id', suffixes=('_orig', '_new'))
    rows = []
    for col in sorted((set(orig.columns) & set(new_df.columns)) - skip):
        a = pd.to_numeric(both[f'{col}_orig'], errors='coerce')
        b = pd.to_numeric(both[f'{col}_new'], errors='coerce')
        m = a.notna() & b.notna()
        if m.sum() < 20 or a[m].nunique() < 2:
            continue
        rows.append({'feature': col,
                     'spearman_orig_vs_harmonised': round(a[m].corr(b[m], method='spearman'), 4),
                     'missing_orig_%': round(a.isna().mean() * 100, 1),
                     'missing_new_%': round(b.isna().mean() * 100, 1)})
    if not rows:
        print("\nToo few tracks to compare with the original features yet.")
        return
    shift = pd.DataFrame(rows).sort_values('spearman_orig_vs_harmonised')
    shift.to_csv(REVISION_DIR / f'feature_shift_{mode}.csv', index=False, encoding='utf-8-sig')
    print("\nFeatures most changed by harmonisation (low Spearman = codec/level-sensitive):")
    print(shift.head(15).to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['common', 'matched'], required=True)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument('--limit', type=int, default=0, help='process only N tracks (quick test)')
    parser.add_argument('--per-cell', type=int, default=0,
                        help='process a balanced random subset: N tracks per language x source cell '
                             '(same tracks every run; e.g. 100 = 600 tracks)')
    args = parser.parse_args()

    harm = pd.read_csv(REVISION_DIR / f'harmonised_{args.mode}.csv')
    harm = harm.loc[harm['ok'].astype(str).str.lower() == 'true']
    orig = pd.read_csv(AUDIO_CSV, low_memory=False)
    orig['song_id'] = orig['song_id'].astype(str).str.strip()
    meta = orig.loc[true_mask(orig['ok']), ['song_id', 'title', 'artist', 'language',
                                            'final_genre', 'source', 'audio_path']] \
        .drop_duplicates('song_id', keep='last').rename(columns={'audio_path': 'original_audio_path'})

    if args.per_cell:
        sel = harm.merge(meta[['song_id', 'language', 'source']], on='song_id')
        harm = select_per_cell(sel, args.per_cell)[['song_id', 'audio_path']]
        print(f'Balanced subset: {len(harm)} tracks ({args.per_cell} per language x source cell)')

    out_csv = OUTPUT_DIR / f'audio_features_{args.mode}.csv'
    done = pd.DataFrame()
    todo = harm[['song_id', 'audio_path']]
    if out_csv.exists():
        done = pd.read_csv(out_csv, low_memory=False)
        finished = set(done.loc[true_mask(done['ok']), 'song_id'].astype(str))
        todo = todo[~todo['song_id'].astype(str).isin(finished)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"Mode: {args.mode} | to extract: {len(todo)} | workers: {args.workers}")

    records = []
    import time
    t_start = time.time()

    def save():
        new = pd.concat([done, pd.DataFrame(records)], ignore_index=True) \
            .drop_duplicates('song_id', keep='last')
        new = meta.merge(new.drop(columns=[c for c in meta.columns if c in new.columns and c != 'song_id']),
                         on='song_id', how='inner')
        new.to_csv(out_csv, index=False, encoding='utf-8-sig')
        return new

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(extract_all, r.song_id, r.audio_path) for r in todo.itertuples(index=False)]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            records.append(rec)
            if not rec['ok']:
                print(f"  ERROR {rec['song_id']}: {rec['error'][:150]}")
            if n % 25 == 0 or n == len(futures):
                el = time.time() - t_start
                print(f"  {n}/{len(futures)} done | {el / 60:.1f} min elapsed | about {el / n * (len(futures) - n) / 60:.0f} min left")
                save()

    new = save()
    print(f"\nSaved {len(new)} rows -> {out_csv}")
    print(new.groupby(['language', 'source']).size().to_string())
    compare_with_original(new, args.mode)
    print("\nNext: point AUDIO_CSV in the fusion notebook to this file to rerun the experiments.")


if __name__ == '__main__':
    main()
