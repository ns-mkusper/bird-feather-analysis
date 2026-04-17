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
        # Import lazily so producer-only environments (for example Nebari notebooks)
        # can dispatch tasks without installing worker-only model dependencies.
        from src.feather_processing import FeatherProcessor

        _PROCESSOR = FeatherProcessor()
    return _PROCESSOR


@celery_app.task(bind=True, name="src.celery_tasks.process_image", autoretry_for=(Exception,), retry_backoff=True, retry_kwargs={"max_retries": 2})
def process_image(self, image_path: str, output_dir: str) -> dict:
    processor = _get_processor()
    result = processor.process_image(image_path=image_path, output_dir=output_dir)
    return {
        "image_path": result.image_path,
        "success": result.success,
        "reason": result.reason,
        "worker": os.uname().nodename,
    }
