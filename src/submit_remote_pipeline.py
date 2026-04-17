from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from typing import Sequence

from celery import group

from src.celery_tasks import process_image


def _list_remote_images(
    host: str,
    user: str,
    key_path: str | None,
    remote_input_dir: str,
    max_images: int | None,
) -> list[str]:
    ssh_cmd: list[str] = ["ssh"]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    ssh_cmd.extend(["-o", "StrictHostKeyChecking=no", f"{user}@{host}"])

    remote_dir_q = shlex.quote(remote_input_dir)
    remote_cmd = (
        f"find {remote_dir_q} -maxdepth 1 -type f "
        r"\( -iname '*.jpg' -o -iname '*.jpeg' \) | sort"
    )
    out = subprocess.check_output([*ssh_cmd, remote_cmd], text=True)
    paths = [line.strip() for line in out.splitlines() if line.strip()]
    if max_images is not None:
        paths = paths[:max_images]
    return paths


def _dispatch(paths: Sequence[str], remote_output_dir: str, poll_seconds: float) -> None:
    if not paths:
        print("No remote JPG/JPEG files found.")
        return

    print(f"Dispatching {len(paths)} image tasks to Celery workers...")
    async_result = group(process_image.s(path, remote_output_dir) for path in paths).apply_async()
    total = len(paths)

    while not async_result.ready():
        print(f"Progress: {async_result.completed_count()}/{total} complete")
        time.sleep(poll_seconds)

    results = async_result.get(disable_sync_subtasks=False)
    successes = sum(1 for item in results if item.get("success"))
    failures = [item for item in results if not item.get("success")]
    print(f"Pipeline complete: {successes}/{total} successful")
    if failures:
        print(f"Failures: {len(failures)}")
        for item in failures[:20]:
            print(f" - {item.get('image_path')}: {item.get('reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dispatch remote image paths to Celery workers without local dataset copies."
    )
    parser.add_argument("--host", required=True, help="Cluster head host/IP for SSH path discovery")
    parser.add_argument("--user", default="openteams", help="SSH user")
    parser.add_argument("--key-path", default="", help="Optional SSH private key path")
    parser.add_argument("--remote-input-dir", required=True, help="Input dir path as seen on cluster nodes")
    parser.add_argument("--remote-output-dir", required=True, help="Output dir path as seen on cluster nodes")
    parser.add_argument("--max-images", type=int, default=None, help="Optional cap on number of images")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Status polling interval")
    args = parser.parse_args()

    paths = _list_remote_images(
        host=args.host,
        user=args.user,
        key_path=args.key_path.strip() or None,
        remote_input_dir=args.remote_input_dir,
        max_images=args.max_images,
    )
    print(f"Found {len(paths)} remote images on {args.host}:{args.remote_input_dir}")
    _dispatch(paths=paths, remote_output_dir=args.remote_output_dir, poll_seconds=args.poll_seconds)


if __name__ == "__main__":
    main()
