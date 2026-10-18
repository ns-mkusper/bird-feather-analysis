from __future__ import annotations

import os
from typing import TYPE_CHECKING

from src.celery_app import celery_app

if TYPE_CHECKING:
    from src.feather_processing import FeatherProcessor


_PROCESSOR: FeatherProcessor | None = None


def _get_processor() -> FeatherProcessor:
    global _PROCESSOR
    if _PROCESSOR is None:
        # Import lazily so producer-only notebook/control-plane environments
        # can dispatch tasks without installing worker-only model dependencies.
        from src.feather_processing import FeatherProcessor

        _PROCESSOR = FeatherProcessor()
    return _PROCESSOR


_LABEL_CLIENT = None


def _label_client():
    """The model for label-only tasks: the worker's loaded MLX model if the
    full processor is already up (or FEATHER_LABEL_VLM=local), otherwise the
    VLM server, so a label-only run does not load SAM on every worker."""
    global _LABEL_CLIENT
    if _PROCESSOR is not None or os.getenv("FEATHER_LABEL_VLM", "auto") == "local":
        return _get_processor().label_client()
    if _LABEL_CLIENT is None:
        from src import label_extract as lx

        _LABEL_CLIENT = lx.VLMClient(lx.load_config()["vlm"])
    return _LABEL_CLIENT


@celery_app.task(bind=True, name="src.celery_tasks.process_image", autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2})
def process_image(self, image_path: str, output_dir: str, options: dict | None = None) -> dict:
    """`options`: {'segmentation': {...}} overrides configs/segmentation.json
    for this task (sample runs try settings without redeploying)."""
    processor = _get_processor()
    # Label table first, from the original slide; cached per output dir, so a
    # retry of this task does not ask the model again.
    labels = processor.extract_labels(image_path, output_dir)
    result = processor.process_image(image_path=image_path, output_dir=output_dir, labels=labels, options=options)
    if labels is not None:
        # cache the feather IDs with the slide's label result
        from src import label_extract as lx

        lx.save_pipeline_result(labels, output_dir)
    return {
        "image_path": result.image_path,
        "success": result.success,
        "reason": result.reason,
        "worker": os.uname().nodename,
        "feather_ids": result.feather_ids,
        "labels": labels.to_json() if labels is not None else None,
    }


@celery_app.task(bind=True, name="src.celery_tasks.extract_labels", autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 1})
def extract_labels(self, image_path: str, output_dir: str) -> dict:
    """Label table only (no segmentation): the cluster side of
    `python -m src.label_extract --distributed`."""
    from src import label_extract as lx

    return lx.extract_for_pipeline(image_path, output_dir, _label_client()).to_json()
