# Saraswati: trilingual AI-music detection, quality assessment and explainability

Code for detecting AI-generated music (Suno v5) against human-made songs in
English, Hindi and Mandarin, using audio features, MuQ-Eval quality scores and
lyric features, plus explainability (SHAP).

> Status: research code accompanying a manuscript under preparation. The numbers
> in the notebooks come from the original (random, track-level) split. The
> `scripts/` folder contains the leakage-safe (artist/lyric-grouped) splits,
> the file-format baseline and the loudness/codec-harmonised re-run added during
> revision. The table below gives those results.

## Main results (Macro-F1 on held-out tracks, Suno v5 vs human)

| Setting | Audio only | Lyrics combined | Fused |
|---|---|---|---|
| Original random split (notebooks, N = 359 test) | 0.9637 | 0.9220 | 0.9749 |
| Grouped split, original audio (N = 362 test) | 0.9862 | 0.9446 | 0.9807 |
| Grouped split, harmonised audio, 365-track subset (N = 73 test) | 0.9722 | 0.8483 | 0.9447 |
| Same 365-track subset, original audio (N = 73 test) | 0.9722 | 0.8483 | 0.9586 |

File-format information alone (codec, bitrate, sample rate, channels) reaches
Macro-F1 = 0.9945 (AUROC 0.9947) on the grouped split, so the original files carry a
strong acquisition signal. Confidence intervals, repeated cross-validation and
shuffled-label controls are in `data/release/results/`.

## Repository layout

| Path | What it is |
|---|---|
| `00_data_collection.ipynb` | Builds the song list and downloads the audio (YouTube/Bilibili for human tracks, Muse and Suno for AI tracks) |
| `01_audio_features.ipynb` | Audio features (librosa, pyloudnorm) and MuQ-Eval quality scores |
| `02_lyrics_analysis.ipynb` | Demucs vocal separation, Whisper transcription, lyric features and multilingual embeddings |
| `03_multimodal_fusion.ipynb` | Audio, lyric and fused classifiers, SHAP and the language analysis |
| `scripts/audit_dataset.py` | Dataset audit: dropped tracks, file-format check, transcript script check |
| `scripts/make_grouped_splits.py` | Artist and near-duplicate-lyric groups, leakage report, grouped splits |
| `scripts/harmonise_audio.py` | Loudness (-14 LUFS) and codec harmonisation |
| `scripts/extract_harmonised_features.py` | Extracts the audio features again from the harmonised audio |
| `scripts/run_grouped_classifiers.py` | Re-runs the five classifiers on the grouped split, with confidence intervals, a file-format-only baseline and a shuffled-label control |
| `scripts/cross_generator_test.py` | Optional test on Udio and older Suno songs from SONICS (not run for the paper) |
| `scripts/export_release_data.py` | Creates `data/release/` (numbers only) |
| `data/release/` | Features, IDs, splits and result tables that can be shared |

`scripts/README.md` explains how to run the scripts.

## Data availability

The audio and lyrics are copyrighted, so they are not included. `data/release/` has only numeric features, song IDs
(public YouTube/Bilibili IDs), split assignments and result tables. With these you can repeat the modelling without the
recordings. To rebuild the audio, run `00_data_collection.ipynb` with your own access to the sources and follow their
terms of service and local law.

## How to reproduce

1. Run `pip install -r requirements.txt` (Python 3.11, with ffmpeg including libmp3lame and libopus on the PATH). A GPU helps for Whisper, Demucs and MuQ-Eval.
2. Run the notebooks in order `00` to `03`, or only `03_multimodal_fusion.ipynb` with the tables in `data/release/`.
3. For the extra checks, run the scripts in `scripts/` in the order given in `scripts/README.md`.

The paths at the top of each notebook (`PROJECT_ROOT`, `OUTPUT_DIR` and so on) are the only settings to edit. API keys are
read at run time and are not stored in the notebooks.

## Known limitations

- Only one generator (Suno v5) is used for training. `scripts/cross_generator_test.py` is there to test Udio and older Suno versions from SONICS, but I have not run it for the paper.
- The human and AI audio were collected in different ways (YouTube MP3 for human, Suno or Muse MP3 for AI). File format alone separates the two classes (see `audit_dataset.py`), so the results on the original files may partly come from how the audio was collected. `harmonise_audio.py` and `extract_harmonised_features.py` give a harmonised version. Re-encoding cannot remove traces of earlier lossy processing, and I did not run the matched-chain control (`--mode matched`).
- The harmonised experiment uses a 365-track subset with 73 test tracks, so its confidence intervals are wide.
- The original random split can put the same artist or near-duplicate lyrics in both train and test. `make_grouped_splits.py` measures this and gives the grouped splits.
- Whisper often labels Hindi as Urdu even when the transcript is in Devanagari.
- Two helper functions were missing from my saved notebooks and I rebuilt them: `transcribe()` in `02_lyrics_analysis.ipynb` and `issue_label()` in `03_multimodal_fusion.ipynb`. They follow how the results are used later, but they are not the original code. Both are marked with a NOTE in the notebooks.

## Data sources and credits

- AI tracks (900): 300 English and 243 Mandarin tracks come from the Muse dataset (Hugging Face `bolshyC/Muse`, Suno v5 provenance); the remaining 57 Mandarin and all 300 Hindi tracks were generated by me with Suno v5. Muse is released under the MIT licence; its audio is not redistributed here. (In `tracks.csv` the column `source_dataset` reads "Self-generated" for all 900 AI tracks, which does not distinguish Muse from directly generated tracks.)
- Human tracks (900): 300 English tracks are listed in the SONICS `real_songs.csv` metadata (CC BY-NC 4.0); the Hindi and Mandarin tracks were found through iTunes/Apple Music search metadata, with a few from Kaggle song lists and MusicBrainz. Audio was obtained from YouTube/Bilibili and is not redistributed; `tracks.csv` lists public video IDs and basic metadata only.
- Cross-generator test (optional): SONICS (CC BY-NC 4.0), downloaded at run time and not redistributed here.

## Citation

If you use this code, please cite the accompanying paper (details to be added on publication).

## Licence

Code: MIT (see `LICENSE`). Released data tables in `data/release/`: CC BY-NC 4.0, because they include metadata derived from SONICS (CC BY-NC 4.0). Song titles, artists and recording rights belong to their owners.
