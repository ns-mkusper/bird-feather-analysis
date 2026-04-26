import argparse
import json
import subprocess
from datetime import datetime, timezone


def _safe_node_label(node_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in node_id)


def _fetch_node_state(user: str, node: str, metrics_dir: str, run_id: str, key_path: str | None) -> dict | None:
    node_label = _safe_node_label(node)
    state_path = f"{metrics_dir}/{run_id}_{node_label}.json"
    alt_state_path = f"~/Feather_Molt_Project/data/runs/{run_id}/pipeline_metrics/{run_id}_{node_label}.json"
    ssh_cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]
    if key_path:
        ssh_cmd.extend(["-i", key_path])
    ssh_cmd.append(f"{user}@{node}")
    ssh_cmd.append(f"if [ -f '{state_path}' ]; then cat '{state_path}'; elif [ -f '{alt_state_path}' ]; then cat '{alt_state_path}'; else exit 1; fi")
    try:
        out = subprocess.check_output(ssh_cmd, text=True, stderr=subprocess.DEVNULL)
        return json.loads(out)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate live pipeline status from node-local metric files")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--nodes", default="10.0.0.148,10.0.0.63,10.0.0.19,10.0.0.118")
    parser.add_argument("--user", default="cluster_user")
    parser.add_argument("--metrics-dir", default="~/Feather_Molt_Project/data/pipeline_metrics")
    parser.add_argument("--key-path", default="")
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()

    node_list = [n.strip() for n in args.nodes.split(",") if n.strip()]
    states = []
    for node in node_list:
        state = _fetch_node_state(
            user=args.user,
            node=node,
            metrics_dir=args.metrics_dir,
            run_id=args.run_id,
            key_path=args.key_path or None,
        )
        if state is None:
            states.append({"node_id": node, "missing": True})
        else:
            states.append(state)

    totals = {
        "processed": 0,
        "success": 0,
        "failed": 0,
        "feathers_saved": 0,
        "duration_ms_sum": 0.0,
        "vlm_score_sum": 0.0,
        "vlm_score_count": 0,
        "selected_count": 0,
        "complete_nodes": 0,
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
    }
    for st in states:
        if st.get("missing"):
            continue
        totals["processed"] += int(st.get("processed", 0))
        totals["success"] += int(st.get("success", 0))
        totals["failed"] += int(st.get("failed", 0))
        totals["feathers_saved"] += int(st.get("feathers_saved", 0))
        totals["duration_ms_sum"] += float(st.get("duration_ms_sum", 0.0))
        totals["vlm_score_sum"] += float(st.get("vlm_score_sum", 0.0))
        totals["vlm_score_count"] += int(st.get("vlm_score_count", 0))
        totals["selected_count"] += int(st.get("selected_count", 0))
        totals["complete_nodes"] += int(st.get("complete", 0))
        totals["metadata_bird_known"] += int(st.get("metadata_bird_known", 0))
        totals["metadata_date_known"] += int(st.get("metadata_date_known", 0))
        totals["metadata_full_known"] += int(st.get("metadata_full_known", 0))
        totals["metadata_unknown_any"] += int(st.get("metadata_unknown_any", 0))
        totals["metadata_from_vlm_fallback"] += int(st.get("metadata_from_vlm_fallback", 0))
        totals["retry_used"] += int(st.get("retry_used", 0))
        totals["retry_selected"] += int(st.get("retry_selected", 0))
        totals["vlm_all_feathers_covered_true"] += int(st.get("vlm_all_feathers_covered_true", 0))
        totals["vlm_all_feathers_covered_count"] += int(st.get("vlm_all_feathers_covered_count", 0))
        totals["vlm_background_leakage_detected"] += int(st.get("vlm_background_leakage_detected", 0))
        totals["vlm_background_leakage_count"] += int(st.get("vlm_background_leakage_count", 0))
        totals["vlm_grouped_boxes_detected"] += int(st.get("vlm_grouped_boxes_detected", 0))
        totals["vlm_grouped_boxes_count"] += int(st.get("vlm_grouped_boxes_count", 0))

    processed = totals["processed"]
    vlm_count = totals["vlm_score_count"]
    result = {
        "run_id": args.run_id,
        "queried_at": datetime.now(timezone.utc).isoformat(),
        "totals": {
            "selected_count": totals["selected_count"],
            "processed": processed,
            "success": totals["success"],
            "failed": totals["failed"],
            "feathers_saved": totals["feathers_saved"],
            "avg_ms_per_image": (totals["duration_ms_sum"] / processed) if processed else 0.0,
            "avg_feathers_per_processed": (totals["feathers_saved"] / processed) if processed else 0.0,
            "vlm_score_avg": (totals["vlm_score_sum"] / vlm_count) if vlm_count else None,
            "vlm_score_count": vlm_count,
            "metadata_bird_known": totals["metadata_bird_known"],
            "metadata_date_known": totals["metadata_date_known"],
            "metadata_full_known": totals["metadata_full_known"],
            "metadata_unknown_any": totals["metadata_unknown_any"],
            "metadata_from_vlm_fallback": totals["metadata_from_vlm_fallback"],
            "retry_used": totals["retry_used"],
            "retry_selected": totals["retry_selected"],
            "retry_used_rate": (totals["retry_used"] / processed) if processed else 0.0,
            "retry_selected_rate": (totals["retry_selected"] / processed) if processed else 0.0,
            "metadata_full_known_rate": (totals["metadata_full_known"] / processed) if processed else 0.0,
            "metadata_unknown_any_rate": (totals["metadata_unknown_any"] / processed) if processed else 0.0,
            "vlm_all_feathers_covered_true": totals["vlm_all_feathers_covered_true"],
            "vlm_all_feathers_covered_count": totals["vlm_all_feathers_covered_count"],
            "vlm_all_feathers_covered_rate": (
                totals["vlm_all_feathers_covered_true"] / totals["vlm_all_feathers_covered_count"]
                if totals["vlm_all_feathers_covered_count"]
                else None
            ),
            "vlm_background_leakage_detected": totals["vlm_background_leakage_detected"],
            "vlm_background_leakage_count": totals["vlm_background_leakage_count"],
            "vlm_background_leakage_rate": (
                totals["vlm_background_leakage_detected"] / totals["vlm_background_leakage_count"]
                if totals["vlm_background_leakage_count"]
                else None
            ),
            "vlm_grouped_boxes_detected": totals["vlm_grouped_boxes_detected"],
            "vlm_grouped_boxes_count": totals["vlm_grouped_boxes_count"],
            "vlm_grouped_boxes_rate": (
                totals["vlm_grouped_boxes_detected"] / totals["vlm_grouped_boxes_count"]
                if totals["vlm_grouped_boxes_count"]
                else None
            ),
            "complete_nodes": totals["complete_nodes"],
            "total_nodes": len(node_list),
        },
        "nodes": states,
    }

    if args.pretty:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(json.dumps(result, separators=(",", ":"), sort_keys=True))


if __name__ == "__main__":
    main()
