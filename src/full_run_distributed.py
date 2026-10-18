import argparse
import json
import os
import time
from glob import glob

from celery import group

from src.celery_tasks import process_image


def _write_labels(results: list[dict], label_dir: str) -> None:
    """Specimen label table for the run, from the label results the workers
    returned with each task (independent of segmentation success)."""
    from pathlib import Path

    from src import label_extract as lx

    labels = [lx.LabelResult.from_json(item["labels"]) if item.get("labels")
              else lx.bare_result(item["image_path"], item["feather_ids"])
              for item in results if item.get("labels") or item.get("feather_ids")]
    if not labels:
        return
    out = Path(label_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "labels.jsonl", "a") as fh:
        for r in labels:
            fh.write(json.dumps(r.to_json()) + "\n")
    summary = lx.finish_run(out, [r.image for r in labels], lx.load_config())
    print(f"Label table: {len(labels)} slides, {summary['needs_review']} need review, "
          f"{summary['slides_with_model_error']} with model errors; per-feather table "
          f"{summary['feathers']} feathers -> {out / 'feathers.csv'}")


def run_pipeline(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    image_paths = sorted(glob(os.path.join(input_dir, "*.[jJ][pP][gG]")))
    if not image_paths:
        print(f"No JPG files found in {input_dir}")
        return

    print(f"Dispatching {len(image_paths)} image tasks to Celery workers...")
    job = group(process_image.s(path, output_dir) for path in image_paths)
    async_result = job.apply_async()

    total = len(image_paths)
    while not async_result.ready():
        completed = async_result.completed_count()
        print(f"Progress: {completed}/{total} complete")
        time.sleep(5)

    results = async_result.get(disable_sync_subtasks=False)
    _write_labels(results, os.path.join(output_dir, "labels"))
    successes = sum(1 for item in results if item.get("success"))
    failures = [item for item in results if not item.get("success")]

    print(f"Pipeline complete: {successes}/{total} successful")
    if failures:
        print(f"Failures: {len(failures)}")
        for item in failures[:20]:
            print(f" - {item.get('image_path')}: {item.get('reason')}")


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Distributed feather processing with Celery")
    parser.add_argument("--input-dir", default=os.path.join(base_dir, "data", "raw"))
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "data", "processed"))
    args = parser.parse_args()
    run_pipeline(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
