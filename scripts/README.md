# Scripts for the extra checks

These scripts hold the extra checks I added after the first version of the study: grouped splits to stop
artist and lyric leakage, a test using only file-format information, and a re-run on audio that has been put
through the same loudness and codec chain. The numbers in the paper's last results section come from them.

Copy the folder next to `outputs/` and `dataset_collection_clean/` and run everything from that project folder
(for example `python scripts/audit_dataset.py`). The scripts expect the feature files made by the four notebooks.
Results go to `outputs/revision/`. If your folder is still called `outputs/week1` from an earlier run, the scripts use it.

| Step | Command | What it does | Time |
|---|---|---|---|
| 1 | `python scripts/audit_dataset.py` | Explains the 9 dropped tracks, tests whether file format alone predicts AI vs human, and checks the script of every transcript | 2-5 min |
| 2 | `python scripts/make_grouped_splits.py` | Builds artist and near-duplicate-lyric groups, measures leakage in the original split and writes the grouped splits | 1-2 min |
| 3 | `python scripts/harmonise_audio.py --mode common` | Puts every track through -14 LUFS and 128 kbps MP3 (about 7 GB). `--mode matched` uses Opus 128k then MP3 192k (about 10 GB) | 10-40 min |
| 4 | `python scripts/extract_harmonised_features.py --mode common` | Extracts the audio features again from the harmonised files | slow, about 100 s per track per worker |
| 5 | `python scripts/cross_generator_test.py --n 10` | Tests the audio model on other generators (Udio and older Suno versions from SONICS) | minutes |
| 6 | `python scripts/run_grouped_classifiers.py --quick --audio original` | Trains the five classifiers on the grouped split and prints test F1, AUROC, confidence intervals, the file-format baseline, a shuffled-label control and repeated grouped CV | 5-10 min |

Run step 6 again with `--audio common` (or `--audio matched`) for the harmonised files. To compare on the same
tracks, add `--only-ids-from outputs/audio_features_common.csv` to the original-audio run.
Steps 3 and 4 can be tried first with `--limit 10`, and they carry on where they stopped if interrupted.
`--workers N` sets the number of processes.

I ran steps 1, 2, 3 (common), 4 (common) and 6. Step 5 and the matched chain (`--mode matched`) have not been run.

## Requirements

ffmpeg and ffprobe on the PATH with `libmp3lame` and `libopus`, and the packages in `requirements.txt`.
The original audio and CSVs are never changed. New files go to `outputs/revision/`,
`outputs/audio_features_<mode>.csv` and `dataset_harmonised/<mode>/`.

## What to look at by eye

- `outputs/revision/near_duplicate_pairs.csv`: check that the linked pairs really are the same lyrics.
  If unrelated songs are linked, raise `LYRIC_JACCARD_THRESHOLD` in step 2. If obvious duplicates are missed, lower it.
- The "Largest group" line in step 2. If one group holds more than about 5% of the data, the artist grouping may need a look.
- `outputs/revision/feature_shift_<mode>.csv`: features with a low Spearman value changed most when only the codec
  and level changed, so they are sensitive to how the audio was acquired.

## Using the grouped split in the fusion notebook

Replace the split cell with:

```python
splits = pd.read_csv('outputs/revision/splits.csv').set_index('song_id')
split_of = merged['song_id'].map(splits['holdout_split']).values
groups_all = merged['song_id'].map(splits['group_id']).values
train_idx = np.where(split_of == 'train')[0]
val_idx = np.where(split_of == 'val')[0]
test_idx = np.where(split_of == 'test')[0]
```

and make the inner tuning grouped: in `tune_model`, use
`StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)` and call
`search.fit(X_train, y_train, groups=groups_all[train_idx])`.
To use harmonised features, set `AUDIO_CSV = OUTPUT_DIR/'audio_features_common.csv'` (or `_matched`).

## Cross-generator test (step 5)

    python scripts/cross_generator_test.py --n 300                                  # Udio (SONICS)
    python scripts/cross_generator_test.py --generator suno --n 100                 # older Suno versions (SONICS)
    python scripts/cross_generator_test.py --generator suno --algorithm v3 --n 100  # only versions containing "v3"
    python scripts/cross_generator_test.py --own-dir D:\new_suno --own-name suno_new   # your own generated files

Each run saves `outputs/revision/<name>_cross_generator_results.csv`. The SONICS modes need
`pip install huggingface_hub remotezip`. The audio is in 10 zip files (32 GB) but only the chosen songs are downloaded.
SONICS licence: CC BY-NC 4.0 (research use).
