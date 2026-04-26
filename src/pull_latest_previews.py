import argparse
import csv
import glob
import json
import os
import subprocess


def _read_output_subdir(repo_root: str, run_id: str) -> str:
    meta_path = os.path.join(repo_root, "data", "runs", f"{run_id}.meta.csv")
    if os.path.exists(meta_path):
        with open(meta_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
            if rows and rows[0].get("output_subdir"):
                return rows[0]["output_subdir"]
    return f"data/runs/{run_id}/processed"


def _safe_node(node: str) -> str:
    return node.replace(".", "_")


def _ssh_base_cmd(key_path: str) -> list[str]:
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=no"]
    if key_path:
        cmd.extend(["-i", key_path])
    return cmd


def _scp_base_cmd(key_path: str) -> list[str]:
    cmd = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=6", "-o", "StrictHostKeyChecking=no"]
    if key_path:
        cmd.extend(["-i", key_path])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Pull latest feather previews from each node for a run")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--nodes", default="10.0.0.148,10.0.0.63,10.0.0.19,10.0.0.118")
    parser.add_argument("--user", default="cluster_user")
    parser.add_argument("--per-node", type=int, default=3)
    parser.add_argument("--max-show", type=int, default=12)
    parser.add_argument("--key-path", default="")
    args = parser.parse_args()

    repo_root = os.path.abspath(args.repo_root)
    output_subdir = _read_output_subdir(repo_root, args.run_id)
    preview_dir = os.path.join(repo_root, "data", "runs", args.run_id, "live_preview")
    os.makedirs(preview_dir, exist_ok=True)

    nodes = [n.strip() for n in args.nodes.split(",") if n.strip()]
    pulled: list[str] = []

    for node in nodes:
        remote_dir = f"~/Feather_Molt_Project/{output_subdir}"
        ssh_cmd = _ssh_base_cmd(args.key_path)
        ssh_cmd.append(f"{args.user}@{node}")
        ssh_cmd.append(f"ls -t '{remote_dir}'/*.jpg 2>/dev/null | head -n {args.per_node}")
        try:
            out = subprocess.check_output(ssh_cmd, text=True, stderr=subprocess.DEVNULL)
            remote_paths = [line.strip() for line in out.splitlines() if line.strip()]
        except Exception:
            remote_paths = []
        for remote_path in remote_paths:
            dst = os.path.join(preview_dir, f"{_safe_node(node)}__{os.path.basename(remote_path)}")
            scp_cmd = _scp_base_cmd(args.key_path)
            scp_cmd.extend([f"{args.user}@{node}:{remote_path}", dst])
            try:
                subprocess.check_call(scp_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                pulled.append(dst)
            except Exception:
                continue

    files = [p for p in glob.glob(os.path.join(preview_dir, "*.jpg")) if os.path.isfile(p)]
    files.sort(key=os.path.getmtime, reverse=True)
    files = files[: args.max_show]
    print(json.dumps({"run_id": args.run_id, "preview_dir": preview_dir, "files": files}, separators=(",", ":")))


if __name__ == "__main__":
    main()
