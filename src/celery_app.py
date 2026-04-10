import os

from celery import Celery


BROKER_URL = os.getenv("BROKER_URL", "redis://10.0.0.148:6379/0")
RESULT_BACKEND = os.getenv("RESULT_BACKEND", "redis://10.0.0.148:6379/1")

celery_app = Celery("feather_pipeline", broker=BROKER_URL, backend=RESULT_BACKEND)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_track_started=True,
    include=["src.celery_tasks"],
)
