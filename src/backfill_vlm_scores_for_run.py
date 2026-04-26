import argparse
import os
import sqlite3
from pathlib import Path

from src.feather_processing import FeatherProcessor


def _to_int_bool(value) -> int | None:
    if isinstance(value, bool):
        return 1 if value else 0
    return None


def _fallback_notes(score: float | None, covered: bool | None, leakage: bool | None, grouped: bool | None) -> str:
    parts: list[str] = []
    if score is not None and score <= 7.0:
        parts.append("low VLM confidence score")
    if covered is False:
        parts.append("possible missing feather coverage")
    if leakage is True:
        parts.append("background leakage detected")
    if grouped is True:
        parts.append("grouped boxes detected")
    return "; ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill VLM scoring fields for an existing run without re-segmentation")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--db-path", required=True)
    parser.add_argument("--processed-dir", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Force VLM scoring path on for this maintenance task.
    os.environ["FEATHER_ENABLE_VLM"] = "1"
    os.environ["FEATHER_ENABLE_VLM_SCORING"] = "1"
    os.environ["FEATHER_ENABLE_VLM_METADATA"] = "1"

    db_path = str(Path(args.db_path).expanduser())
    processed_dir = Path(args.processed_dir).expanduser()
    if not processed_dir.exists():
        raise RuntimeError(f"Processed directory not found: {processed_dir}")

    conn = sqlite3.connect(db_path, timeout=60)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = cur.execute(
        """
        SELECT run_id, node_id, image_path, vlm_score, vlm_notes
        FROM image_stats
        WHERE run_id = ?
          AND (vlm_score IS NULL OR vlm_notes IS NULL OR trim(vlm_notes) = '')
        ORDER BY image_path
        """,
        (args.run_id,),
    ).fetchall()
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    processor = FeatherProcessor()

    scored = 0
    skipped_missing_overlay = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        stem = Path(row["image_path"]).stem
        overlay = processed_dir / f"{stem}_BoundingBoxes.jpg"
        if not overlay.exists():
            skipped_missing_overlay += 1
            continue
        try:
            judge = processor._vlm_judge(str(overlay))
            score_raw = judge.get("quality_score_1_to_10")
            score = float(score_raw) if isinstance(score_raw, (int, float)) else None
            covered = judge.get("all_feathers_covered") if isinstance(judge.get("all_feathers_covered"), bool) else None
            leakage = (
                judge.get("background_leakage_detected")
                if isinstance(judge.get("background_leakage_detected"), bool)
                else None
            )
            grouped = (
                judge.get("green_boxes_grouped_feathers")
                if isinstance(judge.get("green_boxes_grouped_feathers"), bool)
                else None
            )
            notes = str(judge.get("notes") or "").strip()
            if not notes:
                notes = _fallback_notes(score, covered, leakage, grouped)
            if not notes:
                notes = "VLM returned no explicit quality note."

            cur.execute(
                """
                UPDATE image_stats
                SET vlm_score = ?,
                    vlm_all_feathers_covered = ?,
                    vlm_background_leakage_detected = ?,
                    vlm_grouped_boxes_detected = ?,
                    vlm_notes = ?
                WHERE run_id = ? AND node_id = ? AND image_path = ?
                """,
                (
                    score,
                    _to_int_bool(covered),
                    _to_int_bool(leakage),
                    _to_int_bool(grouped),
                    notes[:1024],
                    row["run_id"],
                    row["node_id"],
                    row["image_path"],
                ),
            )
            scored += 1
            if idx % 25 == 0:
                conn.commit()
                print(f"progress {idx}/{len(rows)} scored={scored} skipped={skipped_missing_overlay} failed={failed}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"failed {stem}: {exc}")

    conn.commit()
    print(
        f"done rows={len(rows)} scored={scored} skipped_missing_overlay={skipped_missing_overlay} failed={failed}"
    )


if __name__ == "__main__":
    main()
