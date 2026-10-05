# HSP / S0001 training provenance

These identifier-only CSVs support subject-level overlap checks for the released
Philosopher's Stone model. They contain deidentified HSP identifiers and technical
split metadata, without clinical labels, demographics, dates, scores, latent
values, or original patient identifiers.

## Files

| File | Rows | Purpose |
| --- | ---: | --- |
| [s0001_released_checkpoint_training_recordings.csv](s0001_released_checkpoint_training_recordings.csv) | 14,548 | S0001 recordings with BIDS identifiers in the reconstructed neural-checkpoint training split. |
| [s0001_conservative_development_subject_exclusions.csv](s0001_conservative_development_subject_exclusions.csv) | 17,595 | Recommended subject exclusion list covering the original paper's S0001/MGH population, including legacy inputs, validation records, and held-out records. |

For a conservative independent evaluation, use the subject exclusion list and
exclude **every session of each matched participant**, even when that session
does not appear in the training-record CSV. The subject list is broader than the
checkpoint's training set; membership does not mean every listed subject was
used to optimize that checkpoint.

## Identifier matching

Read identifier columns as strings to preserve their exact values.

- `site_id`: HSP site identifier (`S0001`).
- `BDSPPatientID`: deidentified patient identifier, without the site prefix.
- `participant_id`: complete BIDS subject identifier, formed as
  `sub-` + `site_id` + `BDSPPatientID`.
- `session_id`: BIDS session identifier (`ses-<number>`), present in the
  training-record CSV.
- `recording_id`: `participant_id` + `_` + `session_id`, without a task suffix
  or file extension, present in the training-record CSV.
- `released_checkpoint_split`: `train` for all rows in the training-record CSV.

Match subjects using `participant_id`, or the pair `site_id` and
`BDSPPatientID`. A recording-only match is insufficient to exclude other nights
from the same participant. The HSP `caisr_training_set` flag describes CAISR,
which is a different model; it is not a Philosopher's Stone membership flag.

## Checkpoint and reconstruction

The audit was performed on 2026-10-05 against `SleepPhilosophersStone.ckpt`:

```text
SHA256: b2a9b8dab3ae8543241d613a80cd6a85a6dfca15573f897a1096861b3915af87
Bytes: 2393981880
Epoch: 8
Global step: 210627
Latent dimension: 1024
Held-out fold: 0
```

The checksum matched the checkpoint distributed by the
[Hugging Face repository](https://huggingface.co/wolfgang-ganglberger/philosophers-stone)
on the audit date. These manifests apply to that checkpoint, rather than to
future checkpoints automatically.

Split membership was **reconstructed**, rather than recovered from an archived
per-record execution log. Reconstruction used the ordered original
`master_mp3_wlt_1hz_100f.csv`, with fold 0 held out and the remaining rows split
using `train_test_split(test_size=0.2, random_state=42)`, stratified by
`stratify_quintile`. Subsequent permutations changed row order only.

The original master contains 36,520 entries. Reconstructed overall counts are
23,403 training, 5,851 validation, and 7,266 test entries; the training count
matches the checkpoint. The 22,683 BIDS-keyed S0001 entries partition into
14,548 training, 3,620 validation, and 4,515 test entries. Identifier and
cohort/fold parity with the original paper predictions, available current-master
subject/session joins, and legacy subject crosswalk agreement were checked.

Original master SHA256:
`802a432b331befad834675a54586df25ff573d69fb8ddb1c75c67ead0f0bfd71`.

## Coverage and limitations

The training-record CSV covers BIDS-keyed entries. An additional 347 legacy MGH
input entries comprise 238 training, 39 validation, and 70 test entries. Their
deidentified subjects were resolved and are included in the conservative subject
list, but their session identifiers have not all been resolved. Consequently,
the training-record CSV alone is not a complete recording-level exclusion list.
Legacy and BIDS-keyed entries can refer to the same participants.

There are 106 subjects with entries in both the reconstructed training/validation
pool and the held-out pool. In addition, scalar brain-health and cognition outputs
use separately fitted heads. The packaged head weights match the original
analysis weights; the head-fitting code saves weights fitted using fold-0
analysis rows. Exact executed head-training membership was not independently
recovered. A neural-checkpoint test assignment therefore does not establish
independence from the complete released scoring pipeline.

The conservative list covers the original paper cohort. It is not an exact
minimal exposure list or an exhaustive ledger of every historical
hyperparameter experiment. I0002 and I0006 are absent from the recovered
training/development inputs; this verifies source-site provenance, without
establishing identity disjointness between hospitals.

## HSP v3 model outputs

The [HSP v3 release documentation](https://bdsp.io/content/hsp/3.0/) describes
brain-health and cognition outputs, but an exact researcher-accessible derivative
download path has not been verified in this audit. The sampled
`*_task-psg_pre.csv` was a questionnaire-style key/value file with no latent or
brain-health/cognition fields. This was a bounded sample check, rather than an
inventory of every session in the release.

Separate internal Philosopher's Stone tables contain the full outputs. Their
internal storage location is not a verified external HSP download location.
The CSVs in this folder provide provenance identifiers, rather than model
predictions or latent vectors.

To generate outputs directly, follow the repository's
[inference instructions](../../README.md#inputs) with compatible C4-M1 EEG
inputs. The default `phi_results.csv` includes `lhl_1` through `lhl_1024`,
`brain_health_score`, `total_cognition_score`, `fluid_cognition_score`, and
`crystallized_cognition_score`; see [Outputs](../../README.md#outputs).

## CSV checksums

```text
3522a510a2b88ff24b85b05b802dd0dd42ec366109e232db312179e42373d4b7  s0001_released_checkpoint_training_recordings.csv
29cd446b5ccd58fd9199d889dcce32674c40679148a42ca0c5570ccd657fbbbb  s0001_conservative_development_subject_exclusions.csv
```
