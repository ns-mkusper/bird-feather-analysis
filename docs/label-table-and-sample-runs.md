# Label table, feather IDs and sample runs

The label table is the second output table: bird ID, species, age, sex and date, read from the original slide. It is produced next to the feather segmentation.

## Label table

Open `notebooks/label_lab.ipynb`, set the parameters in the first cell, run.
Regions, prompts and format rules live in `configs/label_fields.json`; edit and re-run, no
restart. Hand-checked answers in `data/label_gold.csv` give per-field accuracy on every run.
CLI: `pixi run labels -- --sample 10 --seed 1`, all slides: `pixi run labels-all`.
Fixed 30-slide benchmark (spread over ID prefixes and years, scored against the gold file):
`pixi run labels -- --manifest configs/label_sample_30.txt --gold data/label_gold.csv`.
Each field tries its primary crop, an expanded crop, then wider `fallback_regions`; the ID and
date are read twice and disagreements go to review. `file_id` (the file name) and
`bird_id` (the code printed top-center on the label) are separate columns; the ID and year parsed
from the file name are only a cross-check or a labelled `filename_fallback`. `N/A` means the
field is not on the slide; a failed model request is `model_error` and stays blank. A run
directory holds `labels.csv`, `review.csv`, `labels.jsonl` (full detail), `labels.sqlite3`
(`specimen_labels` table), `summary.json` and the evidence crops; rerunning with the same
`--out` skips finished slides and retries model errors.
Feather IDs (`<bird_id>_<year>_<letter>`, A, B, ... left to right, e.g. `A0012H_1996_A`) come
from segmentation: workers assign them to the crops they keep (`specimen_feathers` table);
for a label-only run pass `--segmentation-dir data/sample_runs/<id>` from `pixi run sample-run`.
The red/blue text line under the label boxes and the code above the date are ignored.

## Output file names

Code: `src/output_names.py`. Slide `CE01168 2008-im2346.jpg` gives
`CE01168 2008_BoundingBoxes.jpg` and crops `CE01168_2008_Feather_A.jpg`, `..._B.jpg`, ... (A = leftmost),
from the file name's ID and year. A bird with several slides in one year (63 slides) uses the full
label date instead of the year, in file names and feather IDs (`BB17692_19980612_Feather_A.jpg`,
`BB17692H_19980612_A`); slides of the same bird from the same day also get the image number
(`A0514_20000521-im1163_Feather_A.jpg`), as do slides whose own date cannot be read.

A piece that is not a feather gets no letter: when the mask clean-up marks it small or narrow and
the per-feather check (confirmed on a second look) calls it a fragment or not a single feather, it is
saved as `CE01168_2008_Fragment_1.jpg` with its QC image, and listed in `feathers.csv` as a review row
(`segment_status` `fragment`, no feather ID). The letters skip it.

## Label statuses and slide columns

- **`partly_visible`**: the model saw the text only partly. For **bird ID and date** (the feather IDs are built
  from them) that is never readable, whatever the model's confidence: it reported 0.85-0.95 for a bird ID with its
  prefix covered, and confidence is 1.0 on almost every fully visible reading, so it is no guard. **Species and sex**
  (short, low-variance text) accept a partly visible reading from confidence 0.6, **age** from 0.8; an accepted
  one is still listed in `label_problems`.
- **`inferred`** (sex, species): the slide's own reading is not usable, and the same bird's other slides (same
  readable bird ID) all agree; the source slides are named in the field's note. Never inferred from an uncertain
  bird ID; age is never inferred (it changes between years).
- Slide columns, also repeated per feather: `label_problems` (every reason to be careful with the label, in words),
  `label_note`, `photos_same_year` / `multiple_photos_same_year` (slides of this bird ID in this year across the
  whole collection, by file name), `handwritten` / `handwritten_fields` (from a check of the label strip).
- `scripts/rebuild_run.py` rebuilds a finished run's crops and tables under the current rules into new folders,
  from its saved masks and label readings, without the cluster or the model.

## Feathers usable for analysis

`feathers.csv` has every feather, fragment and review row. Its `usable` column is `yes` for a feather
with an ID that passed its checks, is not a duplicate, and whose slide's bird ID and date are both readable; otherwise `excluded_because` says why
(a failed check, `duplicate of ...`, `fragment`, a review row). Checks failed on one look only
(`review_uncertain`) and label problems (`label_needs_review`) do not exclude a feather.
`feathers_usable.csv` has only the usable rows; `summary.json` counts `feathers_usable` and `fragments`.

## Duplicate feathers

Code: `src/feather_duplicates.py`. Same-day slides of one bird often repeat a
feather in another order, position or rotation. Each repeat gets `duplicate_of` (the first
occurrence: earliest image number, then letter) in `feathers.csv`, `specimen_feathers` and
`feather_duplicates.csv`. Matching uses feather size plus SIFT keypoints fitted to one
rotation/shift (mirror images too). Post-run on gathered outputs:
`python -m src.feather_duplicates <processed_dir>`. Lookups also find the older
`<slide>_<id>_<date>_Feather_<n>.jpg` names of earlier runs.
The model is served by `pixi run vlm-server`; point `FEATHER_VLM_ENDPOINTS` at it.

## Label table in cluster runs

every `process_image` task (Celery) and every sharded run
reads the label first, from the original slide, and keeps going with segmentation whatever
the label outcome. Workers reuse their loaded Qwen model (`FEATHER_LABEL_VLM=auto`), cache
each slide's result and evidence under `<output_dir>/labels/`, and the run stores one
`specimen_labels` row per slide in `pipeline_metrics/run_stats.sqlite3` (sharded) or writes
`<output_dir>/labels/labels.csv` etc. on the driver (Celery). `labels_*` counters go to the
run's local state, JSONL events and Redis totals; model errors are counted apart from fields
that are not on the slide. Label-only runs on the workers:
`pixi run labels -- --manifest configs/label_sample_30.txt --distributed --gold data/label_gold.csv`.
`FEATHER_ENABLE_LABELS=0` turns it off.

## Sample segmentation runs

`pixi run sample-run -- --n 5` (or `--match "A0012 1996"`) sends
slides through the workers' normal pipeline and copies the outputs to `data/sample_runs/`.
Their run ids start with `test-`.

## Parking and restarting the cluster

**Park** (as done on 2026-10-20): `touch PAUSED` in the repo folder on every node, then stop the workers and
the VLM server; whatever restarts them (watchdog, launchd) brings them back as small placeholders that load
nothing (`src/node.py`). Model weights can then be deleted to free disk; they are not in git.

**Restart:**
1. On every node, update the repo (`git pull`) and put the SAM 3 weights back as `sam3.pt` in the repo root
   (gated: https://huggingface.co/facebook/sam3). Do this first: without `sam3.pt` the workers silently fall
   back to `dino_sam2`.
2. `rm PAUSED` on every node. Within a minute the placeholders start the real worker / VLM server. The VLM
   (`FEATHER_VLM_MODEL`, Qwen3.8-27B 4-bit, about 16 GB) downloads from Hugging Face on first use, so the first
   slide on each node takes a few extra minutes.
3. Check with `cluster_control.worker_status()` (or `notebooks/run_control.ipynb`), then run a small sample
   before a big run. Reruns of slides already processed can reuse their labels:
   `launch_run(names, run_id, reuse_labels_from="<earlier run>")`.

