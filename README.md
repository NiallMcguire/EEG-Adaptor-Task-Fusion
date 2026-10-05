# EEG Subject x Paradigm Task Fusion

A research implementation of subject-paradigm-conditioned LoRA generation for
zero-shot EEG adaptation. The intended model follows Hyper-X: a small generator
conditions on subject, paradigm and layer embeddings to adapt a frozen backbone.

## Current scope

The first implementation covers the **data infrastructure** milestone:

- Installed MOABB loader verification, without downloading recordings.
- OpenBMI offline labelled trials for MI, ERP and SSVEP, indexed by subject,
  paradigm and session.
- Filtering at 0.5-40 Hz, resampling to 250 Hz, and exact 4 s / 1 s / 4 s windows.
- Versioned `.npy` caches with channel names, class mappings and preprocessing
  provenance. No pickle deserialisation is used.
- One OAS resting covariance per subject/session and fold-specific tangent vectors.
- Saved, write-once partitioned-grid and six-fold subject splits.
- Training-only channel normalisation and Riemannian reference fitting.
- Config/seed commands, CSV records, run provenance and a SLURM launcher.

**Real OpenBMI files have not been downloaded or cached yet.** The extraction
path is tested with synthetic MNE recordings. EEGNeX, LoRA, generator training,
few-shot adaptation, metrics and baselines are not implemented yet. There are no
decoding results or published-performance claims.

## Environment

Use Python 3.11 for the planned experiments. The initial tests ran under Python
3.13 because Python 3.11 was unavailable on the development host; Python 3.11
execution remains to be verified.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[data,dev]'
python -m pytest -q
```

The model dependencies are a separate optional extra and are not needed here.
All commands below run from the repository root. Without an editable install,
prefix them with `PYTHONPATH=src` and use `.venv/bin/python`.

## Data commands

Verify the installed loaders without accessing the recordings:

```bash
python -m eeg_task_fusion.cli verify-api --config configs/openbmi.yaml --seed 42
```

Create the split manifest once, then retain it across experiment seeds:

```bash
python -m eeg_task_fusion.cli splits --config configs/openbmi.yaml --seed 42
```

`split_seed` is independent of later training seeds. Existing manifests are never
overwritten or silently regenerated. To intentionally change the split, choose a
new manifest filename and split seed in the config.

Download and extract the raw data explicitly:

```bash
python -m eeg_task_fusion.cli cache --config configs/openbmi.yaml --seed 42 --download
```

This may download large MATLAB files. MOABB manages the download directory;
configure `MNE_DATA` for your storage before running this command. Extraction is
resumable: completed task/rest entries are reused without reloading recordings.
Do not run two extraction jobs against the same cache simultaneously.

After caching both sessions for all configured subjects, prepare all seven splits:

```bash
python -m eeg_task_fusion.cli prepare --config configs/openbmi.yaml --seed 42
```

Each run records its resolved config, seed, package versions, git hash/status and
per-subject CSV in `results/openbmi_data/<run_id>/`. An optional `--run-id` names a
run explicitly; existing run directories cannot be reused. In a repository with
no commits yet, `git_hash` is `null`, not a fabricated revision.

Preparation saves each split, normalisation arrays, reference covariance,
statistic provenance and one tangent vector for every subject/session. These CSVs
report data preparation, **not decoding metrics**. Failed runs retain an error
record and must not be treated as complete artefacts.

## Scientific protocol

- All three installed Lee2019 loaders have subject IDs 1-54 and sessions 1/2.
  The runtime API command checks this explicitly. Verification used MOABB 1.7.2.
- MOABB exposes `resting_state=True` and named rest runs. The implementation
  requires `1train` and `0preTrainRest` when a real recording is loaded and fails
  explicitly if the installed loader produces different keys. Run-key contents
  have not yet been verified on a downloaded recording.
- Session 2 evaluation uses the **offline labelled run of session 2**, not MOABB's
  `4test` online run. MOABB warns that MI/SSVEP online runs lack labels. ERP also
  uses only the offline run, consistently across paradigms.
- Labels are stable: MI left/right = 0/1; ERP non-target/target = 0/1; SSVEP
  5.45/6.67/8.57/12.0 Hz = 0/1/2/3. Stimulus and EMG channels are excluded.
- Rest uses the first 30 s of **MI pre-training rest** for both sessions, without
  task trials or labels. This is an explicit initial design choice, not an
  aggregation of all paradigm-specific rest. `rest_paradigm` and duration are
  configurable and included in cache identity; short recordings fail explicitly.
- Setting 1 reserves nine validation subjects. Each of the other 45 subjects
  loses one paradigm, giving 15 held-out pairs per paradigm. Training and test
  subjects overlap intentionally, but training/test pairs do not.
- Setting 2 has six groups of nine subjects. Each fold uses one test group, the
  next group for validation, and the remaining 36 subjects for training.
- Training and validation use session 1. Test uses session 2. Normalisation is
  pooled per channel/paradigm over **allowed training pairs only**. Tangent-space
  reference fitting uses one session-1 covariance per training subject, equally
  weighted. Validation/test covariances never update the reference.
- The resting vectors for all subjects/sessions are transformed using the fixed
  training reference. No held-out task recordings are read during fold fitting.
- Future backbone pretraining must obey the same allowed pairs as generator
  training; otherwise the unseen-pair experiment would leak labelled data.

Cached trials are accessed with `ArrayCache.get(subject_id, paradigm_id, session)`
and expose `X`, `y`, channel names and metadata. `FoldDataset` wraps this interface
and applies persisted training statistics on demand, without duplicating trial
arrays for every fold. Training callers must still select the saved training
pairs and session; the dataset wrapper is not itself an evaluation sampler.

## SLURM

Create the environment on the cluster and submit from the repository root:

```bash
sbatch --export=ALL,STAGE=cache,SEED=42 scripts/slurm/data.sbatch
sbatch --export=ALL,STAGE=prepare,SEED=42 scripts/slurm/data.sbatch
```

Wait for caching to complete before preparation, or supply an `afterok` job
dependency. `CONFIG` and `PYTHON_BIN` can override the config and interpreter.
The script needs no GPU. Adjust time/memory to the cluster and add its required
partition/account flags at submission.

## Next milestones

1. Cache real OpenBMI and inspect retained trial counts, channel order, rest
   lengths and label distributions before declaring the data milestone complete.
2. Add EEGNeX and functional LoRA with zero-B equivalence/reference tests, then
   shared and per-subject baselines. Compare with published protocols carefully:
   ERP character-level accuracy is not trial-level AUROC.
3. Add the generator and shape/zero-initialisation tests, then run Setting 1.
4. Align resting embeddings to learnt embeddings and validate on genuinely unseen
   subjects with resting-derived embeddings, not unseen lookup-table entries.
5. Run Setting 2, few-shot curves, remaining baselines and ablations.

No model experiment should run before the LoRA, generator and split gates exist
and pass. Five training seeds reuse the same saved split manifest.