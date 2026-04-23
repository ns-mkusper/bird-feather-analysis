import argparse
import json
import os
import sqlite3
import socket
import time
from datetime import datetime, timezone
from glob import glob

import redis

from src.feather_processing import FeatherProcessor


def pick_shard(paths: list[str], shard_index: int, shard_count: int) -> list[str]:
    return [p for i, p in enumerate(paths) if i % shard_count == shard_index]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect_metrics_redis() -> redis.Redis | None:
    host = os.getenv("FEATHER_METRICS_REDIS_HOST", "10.0.0.148")
    port = int(os.getenv("FEATHER_METRICS_REDIS_PORT", "6379"))
    db = int(os.getenv("FEATHER_METRICS_REDIS_DB", "2"))
    try:
        client = redis.Redis(host=host, port=port, db=db, decode_responses=True, socket_timeout=2.0)
        client.ping()
        return client
    except Exception as exc:  # noqa: BLE001
        print(f"metrics redis unavailable ({host}:{port}/{db}): {exc}")
        return None


def _safe_node_label(node_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_id)


def _write_local_state(path: str, state: dict) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(state, handle, sort_keys=True)
    os.replace(tmp, path)


def _append_jsonl(path: str, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")


def _init_stats_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS image_stats (
            run_id TEXT NOT NULL,
            node_id TEXT NOT NULL,
            image_path TEXT NOT NULL,
            ts TEXT NOT NULL,
            success INTEGER NOT NULL,
            reason TEXT,
            feathers_saved INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            vlm_score REAL,
            vlm_all_feathers_covered INTEGER,
            vlm_background_leakage_detected INTEGER,
            vlm_grouped_boxes_detected INTEGER,
            metadata_source TEXT,
            bird_id TEXT,
            date_text TEXT,
            retry_used INTEGER NOT NULL,
            retry_selected INTEGER NOT NULL,
            profile_selected TEXT,
            PRIMARY KEY (run_id, node_id, image_path)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_image_stats_run ON image_stats(run_id)")
    conn.commit()
    return conn


def _upsert_stats_row(conn: sqlite3.Connection, row: dict) -> None:
    conn.execute(
        """
        INSERT INTO image_stats (
            run_id, node_id, image_path, ts, success, reason, feathers_saved, duration_ms,
            vlm_score, vlm_all_feathers_covered, vlm_background_leakage_detected, vlm_grouped_boxes_detected,
            metadata_source, bird_id, date_text, retry_used, retry_selected, profile_selected
        ) VALUES (
            :run_id, :node_id, :image_path, :ts, :success, :reason, :feathers_saved, :duration_ms,
            :vlm_score, :vlm_all_feathers_covered, :vlm_background_leakage_detected, :vlm_grouped_boxes_detected,
            :metadata_source, :bird_id, :date_text, :retry_used, :retry_selected, :profile_selected
        )
        ON CONFLICT(run_id, node_id, image_path) DO UPDATE SET
            ts=excluded.ts,
            success=excluded.success,
            reason=excluded.reason,
            feathers_saved=excluded.feathers_saved,
            duration_ms=excluded.duration_ms,
            vlm_score=excluded.vlm_score,
            vlm_all_feathers_covered=excluded.vlm_all_feathers_covered,
            vlm_background_leakage_detected=excluded.vlm_background_leakage_detected,
            vlm_grouped_boxes_detected=excluded.vlm_grouped_boxes_detected,
            metadata_source=excluded.metadata_source,
            bird_id=excluded.bird_id,
            date_text=excluded.date_text,
            retry_used=excluded.retry_used,
            retry_selected=excluded.retry_selected,
            profile_selected=excluded.profile_selected
        """,
        row,
    )
    conn.commit()


def _emit_run_start(
    client: redis.Redis | None,
    run_id: str,
    node_id: str,
    shard_index: int,
    shard_count: int,
    selected_count: int,
    input_total: int,
) -> None:
    if client is None:
        return
    run_meta_key = f"feather:run:{run_id}:meta"
    node_key = f"feather:run:{run_id}:node:{node_id}"
    node_set_key = f"feather:run:{run_id}:nodes"
    try:
        pipe = client.pipeline()
        pipe.hsetnx(run_meta_key, "started_at", _now_iso())
        pipe.hset(run_meta_key, mapping={
            "run_id": run_id,
            "input_total": input_total,
            "shard_count": shard_count,
            "last_update": _now_iso(),
        })
        pipe.sadd(node_set_key, node_id)
        pipe.hset(node_key, mapping={
            "node_id": node_id,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "selected_count": selected_count,
            "processed": 0,
            "success": 0,
            "failed": 0,
            "feathers_saved": 0,
            "duration_ms_sum": 0.0,
            "vlm_score_sum": 0.0,
            "vlm_score_count": 0,
            "metadata_bird_known": 0,
            "metadata_date_known": 0,
            "metadata_full_known": 0,
            "metadata_unknown_any": 0,
            "metadata_from_vlm_fallback": 0,
            "retry_used": 0,
            "retry_selected": 0,
            "vlm_all_feathers_covered_true": 0,
            "vlm_all_feathers_covered_count": 0,
            "vlm_background_leakage_detected": 0,
            "vlm_background_leakage_count": 0,
            "vlm_grouped_boxes_detected": 0,
            "vlm_grouped_boxes_count": 0,
            "last_metadata_source": "",
            "last_profile": "default",
            "last_reason": "",
            "last_update": _now_iso(),
        })
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"metrics start emit failed: {exc}")


def _emit_step(
    client: redis.Redis | None,
    run_id: str,
    node_id: str,
    success: bool,
    feathers_saved: int,
    duration_ms: float,
    reason: str,
    vlm_score: float | None,
    metadata_source: str,
    bird_known: bool,
    date_known: bool,
    vlm_all_feathers_covered: bool | None,
    vlm_background_leakage_detected: bool | None,
    vlm_green_boxes_grouped_feathers: bool | None,
    retry_used: bool,
    retry_selected: bool,
) -> None:
    if client is None:
        return
    totals_key = f"feather:run:{run_id}:totals"
    node_key = f"feather:run:{run_id}:node:{node_id}"
    run_meta_key = f"feather:run:{run_id}:meta"
    event_key = f"feather:run:{run_id}:events"
    try:
        pipe = client.pipeline()
        pipe.hincrby(totals_key, "processed", 1)
        pipe.hincrby(totals_key, "success", 1 if success else 0)
        pipe.hincrby(totals_key, "failed", 0 if success else 1)
        pipe.hincrby(totals_key, "feathers_saved", int(feathers_saved))
        pipe.hincrbyfloat(totals_key, "duration_ms_sum", float(duration_ms))
        pipe.hincrby(totals_key, "metadata_bird_known", 1 if bird_known else 0)
        pipe.hincrby(totals_key, "metadata_date_known", 1 if date_known else 0)
        pipe.hincrby(totals_key, "metadata_full_known", 1 if (bird_known and date_known) else 0)
        pipe.hincrby(totals_key, "metadata_unknown_any", 1 if (not bird_known or not date_known) else 0)
        pipe.hincrby(totals_key, "metadata_from_vlm_fallback", 1 if metadata_source == "vlm_fallback" else 0)
        pipe.hincrby(totals_key, "retry_used", 1 if retry_used else 0)
        pipe.hincrby(totals_key, "retry_selected", 1 if retry_selected else 0)

        pipe.hincrby(node_key, "processed", 1)
        pipe.hincrby(node_key, "success", 1 if success else 0)
        pipe.hincrby(node_key, "failed", 0 if success else 1)
        pipe.hincrby(node_key, "feathers_saved", int(feathers_saved))
        pipe.hincrbyfloat(node_key, "duration_ms_sum", float(duration_ms))
        pipe.hincrby(node_key, "metadata_bird_known", 1 if bird_known else 0)
        pipe.hincrby(node_key, "metadata_date_known", 1 if date_known else 0)
        pipe.hincrby(node_key, "metadata_full_known", 1 if (bird_known and date_known) else 0)
        pipe.hincrby(node_key, "metadata_unknown_any", 1 if (not bird_known or not date_known) else 0)
        pipe.hincrby(node_key, "metadata_from_vlm_fallback", 1 if metadata_source == "vlm_fallback" else 0)
        pipe.hincrby(node_key, "retry_used", 1 if retry_used else 0)
        pipe.hincrby(node_key, "retry_selected", 1 if retry_selected else 0)
        pipe.hset(node_key, "last_reason", reason)
        pipe.hset(node_key, "last_metadata_source", metadata_source)
        pipe.hset(node_key, "last_profile", "strict_retry" if retry_selected else "default")
        pipe.hset(node_key, "last_update", _now_iso())
        pipe.hset(run_meta_key, "last_update", _now_iso())

        if vlm_score is not None:
            pipe.hincrbyfloat(totals_key, "vlm_score_sum", float(vlm_score))
            pipe.hincrby(totals_key, "vlm_score_count", 1)
            pipe.hincrbyfloat(node_key, "vlm_score_sum", float(vlm_score))
            pipe.hincrby(node_key, "vlm_score_count", 1)
        if vlm_all_feathers_covered is not None:
            pipe.hincrby(totals_key, "vlm_all_feathers_covered_true", 1 if vlm_all_feathers_covered else 0)
            pipe.hincrby(totals_key, "vlm_all_feathers_covered_count", 1)
            pipe.hincrby(node_key, "vlm_all_feathers_covered_true", 1 if vlm_all_feathers_covered else 0)
            pipe.hincrby(node_key, "vlm_all_feathers_covered_count", 1)
        if vlm_background_leakage_detected is not None:
            pipe.hincrby(totals_key, "vlm_background_leakage_detected", 1 if vlm_background_leakage_detected else 0)
            pipe.hincrby(totals_key, "vlm_background_leakage_count", 1)
            pipe.hincrby(node_key, "vlm_background_leakage_detected", 1 if vlm_background_leakage_detected else 0)
            pipe.hincrby(node_key, "vlm_background_leakage_count", 1)
        if vlm_green_boxes_grouped_feathers is not None:
            pipe.hincrby(totals_key, "vlm_grouped_boxes_detected", 1 if vlm_green_boxes_grouped_feathers else 0)
            pipe.hincrby(totals_key, "vlm_grouped_boxes_count", 1)
            pipe.hincrby(node_key, "vlm_grouped_boxes_detected", 1 if vlm_green_boxes_grouped_feathers else 0)
            pipe.hincrby(node_key, "vlm_grouped_boxes_count", 1)

        pipe.xadd(
            event_key,
            {
                "ts": _now_iso(),
                "node_id": node_id,
                "success": int(success),
                "feathers_saved": int(feathers_saved),
                "duration_ms": f"{duration_ms:.1f}",
                "reason": reason or "",
                "vlm_score": "" if vlm_score is None else f"{vlm_score:.5f}",
                "metadata_source": metadata_source,
                "bird_known": int(bird_known),
                "date_known": int(date_known),
                "vlm_all_feathers_covered": "" if vlm_all_feathers_covered is None else int(vlm_all_feathers_covered),
                "vlm_background_leakage_detected": "" if vlm_background_leakage_detected is None else int(vlm_background_leakage_detected),
                "vlm_grouped_boxes": "" if vlm_green_boxes_grouped_feathers is None else int(vlm_green_boxes_grouped_feathers),
                "retry_used": int(retry_used),
                "retry_selected": int(retry_selected),
            },
            maxlen=2000,
            approximate=True,
        )
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"metrics step emit failed: {exc}")


def _emit_run_complete(
    client: redis.Redis | None,
    run_id: str,
    node_id: str,
    ok: int,
    total: int,
) -> None:
    if client is None:
        return
    run_meta_key = f"feather:run:{run_id}:meta"
    node_key = f"feather:run:{run_id}:node:{node_id}"
    try:
        pipe = client.pipeline()
        pipe.hset(run_meta_key, mapping={"last_update": _now_iso(), "ended_at": _now_iso()})
        pipe.hset(node_key, mapping={"complete": 1, "ok_final": ok, "selected_count": total, "last_update": _now_iso()})
        pipe.execute()
    except Exception as exc:  # noqa: BLE001
        print(f"metrics completion emit failed: {exc}")


def run_shard(
    input_dir: str,
    output_dir: str,
    shard_index: int,
    shard_count: int,
    offset: int,
    max_images: int | None,
) -> None:
    all_paths = sorted(glob(os.path.join(input_dir, "*.[jJ][pP][gG]")))
    if offset > 0:
        all_paths = all_paths[offset:]
    if max_images is not None:
        all_paths = all_paths[:max_images]
    selected = pick_shard(all_paths, shard_index, shard_count)
    os.makedirs(output_dir, exist_ok=True)

    run_id = os.getenv("FEATHER_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    node_id = os.getenv("FEATHER_NODE_ID", socket.gethostname())
    metrics_dir = os.getenv("FEATHER_METRICS_DIR", os.path.join(os.path.dirname(output_dir), "pipeline_metrics"))
    os.makedirs(metrics_dir, exist_ok=True)
    local_state_path = os.path.join(metrics_dir, f"{run_id}_{_safe_node_label(node_id)}.json")
    events_jsonl_path = os.path.join(metrics_dir, f"{run_id}_{_safe_node_label(node_id)}_events.jsonl")
    stats_db_path = os.path.join(metrics_dir, "run_stats.sqlite3")
    stats_conn = _init_stats_db(stats_db_path)

    local_state: dict[str, object] = {
        "run_id": run_id,
        "node_id": node_id,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "selected_count": len(selected),
        "input_total": len(all_paths),
        "processed": 0,
        "success": 0,
        "failed": 0,
        "feathers_saved": 0,
        "duration_ms_sum": 0.0,
        "vlm_score_sum": 0.0,
        "vlm_score_count": 0,
        "vlm_score_avg": None,
        "metadata_bird_known": 0,
        "metadata_date_known": 0,
        "metadata_full_known": 0,
        "metadata_unknown_any": 0,
        "metadata_from_vlm_fallback": 0,
        "retry_used": 0,
        "retry_selected": 0,
        "vlm_all_feathers_covered_true": 0,
        "vlm_all_feathers_covered_count": 0,
        "vlm_background_leakage_detected": 0,
        "vlm_background_leakage_count": 0,
        "vlm_grouped_boxes_detected": 0,
        "vlm_grouped_boxes_count": 0,
        "last_metadata_source": "",
        "last_profile": "default",
        "last_reason": "",
        "started_at": _now_iso(),
        "last_update": _now_iso(),
        "complete": 0,
    }
    _write_local_state(local_state_path, local_state)

    metrics_redis = _connect_metrics_redis()
    _emit_run_start(metrics_redis, run_id, node_id, shard_index, shard_count, len(selected), len(all_paths))

    print(f"Shard {shard_index}/{shard_count}: {len(selected)} images (from {len(all_paths)} total)")
    print(
        "Live metrics:",
        json.dumps(
            {
                "run_id": run_id,
                "node_id": node_id,
                "redis": os.getenv("FEATHER_METRICS_REDIS_HOST", "10.0.0.148"),
                "db": int(os.getenv("FEATHER_METRICS_REDIS_DB", "2")),
            }
        ),
    )
    if not selected:
        _emit_run_complete(metrics_redis, run_id, node_id, ok=0, total=0)
        stats_conn.close()
        return

    processor = FeatherProcessor()
    ok = 0

    def qa_penalty(res) -> int:
        penalty = 0
        if res.vlm_green_boxes_grouped_feathers is True:
            # Grouped-box flag is common; penalize hard only when extraction count is low.
            penalty += 2 if int(res.feathers_saved) <= 3 else 1
        if res.vlm_background_leakage_detected is True:
            penalty += 2
        if res.vlm_all_feathers_covered is False:
            penalty += 2
        if res.vlm_score is not None:
            penalty += max(0, int(10 - float(res.vlm_score)))
        return penalty

    def should_retry(res) -> bool:
        if not res.success:
            return False
        low_score = res.vlm_score is not None and float(res.vlm_score) <= 6.5
        low_feather_count = int(res.feathers_saved) <= 4
        return (
            (res.vlm_green_boxes_grouped_feathers is True and (low_score or low_feather_count))
            or res.vlm_background_leakage_detected is True
            or res.vlm_all_feathers_covered is False
        )

    for idx, path in enumerate(selected, start=1):
        t0 = time.perf_counter()
        result = processor.process_image(path, output_dir, profile="default")
        retry_used = False
        retry_selected = False
        if should_retry(result):
            retry_used = True
            retry_result = processor.process_image(path, output_dir, profile="strict_retry")
            if retry_result.success:
                base_penalty = qa_penalty(result)
                retry_penalty = qa_penalty(retry_result)
                choose_retry = False
                if retry_penalty < base_penalty:
                    choose_retry = True
                elif retry_penalty == base_penalty:
                    if retry_result.vlm_score is not None and result.vlm_score is not None:
                        choose_retry = float(retry_result.vlm_score) >= float(result.vlm_score)
                    else:
                        choose_retry = True
                if choose_retry:
                    result = retry_result
                    retry_selected = True
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result.success:
            ok += 1

        local_state["processed"] = int(local_state["processed"]) + 1
        local_state["success"] = int(local_state["success"]) + (1 if result.success else 0)
        local_state["failed"] = int(local_state["failed"]) + (0 if result.success else 1)
        local_state["feathers_saved"] = int(local_state["feathers_saved"]) + int(result.feathers_saved)
        local_state["duration_ms_sum"] = float(local_state["duration_ms_sum"]) + float(elapsed_ms)
        local_state["last_reason"] = result.reason
        local_state["last_metadata_source"] = result.metadata_source
        local_state["last_update"] = _now_iso()
        bird_known = result.bird_id != "UNKNOWN"
        date_known = result.date != "UNKNOWN"
        local_state["metadata_bird_known"] = int(local_state["metadata_bird_known"]) + (1 if bird_known else 0)
        local_state["metadata_date_known"] = int(local_state["metadata_date_known"]) + (1 if date_known else 0)
        local_state["metadata_full_known"] = int(local_state["metadata_full_known"]) + (1 if (bird_known and date_known) else 0)
        local_state["metadata_unknown_any"] = int(local_state["metadata_unknown_any"]) + (1 if (not bird_known or not date_known) else 0)
        local_state["metadata_from_vlm_fallback"] = int(local_state["metadata_from_vlm_fallback"]) + (
            1 if result.metadata_source == "vlm_fallback" else 0
        )
        local_state["retry_used"] = int(local_state["retry_used"]) + (1 if retry_used else 0)
        local_state["retry_selected"] = int(local_state["retry_selected"]) + (1 if retry_selected else 0)
        local_state["last_profile"] = result.processing_profile
        if result.vlm_all_feathers_covered is not None:
            local_state["vlm_all_feathers_covered_true"] = int(local_state["vlm_all_feathers_covered_true"]) + (
                1 if result.vlm_all_feathers_covered else 0
            )
            local_state["vlm_all_feathers_covered_count"] = int(local_state["vlm_all_feathers_covered_count"]) + 1
        if result.vlm_background_leakage_detected is not None:
            local_state["vlm_background_leakage_detected"] = int(local_state["vlm_background_leakage_detected"]) + (
                1 if result.vlm_background_leakage_detected else 0
            )
            local_state["vlm_background_leakage_count"] = int(local_state["vlm_background_leakage_count"]) + 1
        if result.vlm_green_boxes_grouped_feathers is not None:
            local_state["vlm_grouped_boxes_detected"] = int(local_state["vlm_grouped_boxes_detected"]) + (
                1 if result.vlm_green_boxes_grouped_feathers else 0
            )
            local_state["vlm_grouped_boxes_count"] = int(local_state["vlm_grouped_boxes_count"]) + 1
        if result.vlm_score is not None:
            local_state["vlm_score_sum"] = float(local_state["vlm_score_sum"]) + float(result.vlm_score)
            local_state["vlm_score_count"] = int(local_state["vlm_score_count"]) + 1
        vlm_count = int(local_state["vlm_score_count"])
        local_state["vlm_score_avg"] = (
            float(local_state["vlm_score_sum"]) / vlm_count if vlm_count else None
        )
        _write_local_state(local_state_path, local_state)
        _append_jsonl(
            events_jsonl_path,
            {
                "ts": _now_iso(),
                "run_id": run_id,
                "node_id": node_id,
                "image_path": result.image_path,
                "success": result.success,
                "reason": result.reason,
                "feathers_saved": result.feathers_saved,
                "vlm_score": result.vlm_score,
                "vlm_all_feathers_covered": result.vlm_all_feathers_covered,
                "vlm_background_leakage_detected": result.vlm_background_leakage_detected,
                "vlm_grouped_boxes_detected": result.vlm_green_boxes_grouped_feathers,
                "metadata_source": result.metadata_source,
                "bird_id": result.bird_id,
                "date": result.date,
                "retry_used": retry_used,
                "retry_selected": retry_selected,
                "profile_selected": result.processing_profile,
            },
        )
        _upsert_stats_row(
            stats_conn,
            {
                "run_id": run_id,
                "node_id": node_id,
                "image_path": result.image_path,
                "ts": _now_iso(),
                "success": 1 if result.success else 0,
                "reason": result.reason or "",
                "feathers_saved": int(result.feathers_saved),
                "duration_ms": float(elapsed_ms),
                "vlm_score": result.vlm_score,
                "vlm_all_feathers_covered": None if result.vlm_all_feathers_covered is None else (1 if result.vlm_all_feathers_covered else 0),
                "vlm_background_leakage_detected": None if result.vlm_background_leakage_detected is None else (1 if result.vlm_background_leakage_detected else 0),
                "vlm_grouped_boxes_detected": None if result.vlm_green_boxes_grouped_feathers is None else (1 if result.vlm_green_boxes_grouped_feathers else 0),
                "metadata_source": result.metadata_source,
                "bird_id": result.bird_id,
                "date_text": result.date,
                "retry_used": 1 if retry_used else 0,
                "retry_selected": 1 if retry_selected else 0,
                "profile_selected": result.processing_profile,
            },
        )

        _emit_step(
            metrics_redis,
            run_id,
            node_id,
            result.success,
            result.feathers_saved,
            elapsed_ms,
            result.reason,
            result.vlm_score,
            result.metadata_source,
            bird_known,
            date_known,
            result.vlm_all_feathers_covered,
            result.vlm_background_leakage_detected,
            result.vlm_green_boxes_grouped_feathers,
            retry_used,
            retry_selected,
        )
        if idx % 10 == 0 or idx == len(selected):
            print(f"Shard {shard_index}: {idx}/{len(selected)} done, ok={ok}")
    local_state["complete"] = 1
    local_state["ok_final"] = ok
    local_state["ended_at"] = _now_iso()
    local_state["last_update"] = _now_iso()
    _write_local_state(local_state_path, local_state)
    _emit_run_complete(metrics_redis, run_id, node_id, ok=ok, total=len(selected))
    stats_conn.close()
    print(f"Shard {shard_index} complete: ok={ok}/{len(selected)}")


def main() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description="Sharded feather processing without Celery")
    parser.add_argument("--input-dir", default=os.path.join(base_dir, "data", "raw"))
    parser.add_argument("--output-dir", default=os.path.join(base_dir, "data", "processed"))
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    run_shard(args.input_dir, args.output_dir, args.shard_index, args.shard_count, args.offset, args.max_images)


if __name__ == "__main__":
    main()
