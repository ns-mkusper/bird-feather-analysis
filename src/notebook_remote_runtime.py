from __future__ import annotations

import os
import shlex
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from celery import Celery


@dataclass(slots=True)
class RemoteNotebookConfig:
    head_host: str
    ssh_user: str
    key_path: str
    remote_repo_dir: str
    remote_input_dir: str
    remote_output_dir: str
    cluster_hosts: list[str]
    broker_url: str
    result_backend: str


def _first_existing_path(paths: list[str]) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def load_notebook_config_from_env() -> RemoteNotebookConfig:
    head_host = os.getenv("FEATHER_HEAD_HOST", "10.0.0.148")
    ssh_user = os.getenv("FEATHER_SSH_USER", "openteams")

    key_candidates = [
        os.path.expanduser(os.getenv("FEATHER_SSH_KEY", "")),
        os.path.expanduser("~/.ssh/ubuntu-mac-openteams-admin"),
        os.path.expanduser("~/.ssh/ubuntu-mac-cluster_user-admin"),
        os.path.expanduser("~/.ssh/id_ed25519"),
        os.path.expanduser("~/.ssh/llama_watchdog_key"),
    ]
    key_path = _first_existing_path(key_candidates)

    remote_repo_dir = os.getenv("FEATHER_REMOTE_REPO_DIR", "~/Feather_Molt_Project")
    remote_input_dir = os.getenv("FEATHER_REMOTE_INPUT_DIR", f"{remote_repo_dir}/data/raw")
    remote_output_dir = os.getenv("FEATHER_REMOTE_OUTPUT_DIR", f"{remote_repo_dir}/data/processed")

    cluster_hosts = [
        h.strip()
        for h in os.getenv("FEATHER_CLUSTER_HOSTS", "10.0.0.148,10.0.0.63,10.0.0.19,10.0.0.118").split(",")
        if h.strip()
    ]

    broker_url = os.getenv("BROKER_URL", f"redis://{head_host}:6379/0")
    result_backend = os.getenv("RESULT_BACKEND", "")

    return RemoteNotebookConfig(
        head_host=head_host,
        ssh_user=ssh_user,
        key_path=key_path,
        remote_repo_dir=remote_repo_dir,
        remote_input_dir=remote_input_dir,
        remote_output_dir=remote_output_dir,
        cluster_hosts=cluster_hosts,
        broker_url=broker_url,
        result_backend=result_backend,
    )


def assert_tcp_reachable(url: str, name: str, timeout: float = 1.5) -> None:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 6379
    if not host:
        raise RuntimeError(f"{name} URL is invalid: {url}")

    sock = socket.socket()
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"{name} not reachable at {host}:{port} from this notebook host. "
            f"Original error: {type(exc).__name__}: {exc}"
        ) from exc
    finally:
        sock.close()


def create_celery_client(config: RemoteNotebookConfig) -> Celery:
    # Keep env vars in sync so any downstream imports see the same broker settings.
    os.environ["BROKER_URL"] = config.broker_url
    if config.result_backend:
        os.environ["RESULT_BACKEND"] = config.result_backend

    backend = config.result_backend or None
    return Celery("feather_pipeline", broker=config.broker_url, backend=backend)


class RemoteClusterIO:
    def __init__(self, config: RemoteNotebookConfig):
        self.config = config
        self._remote_home_cache: dict[str, str] = {}

    def _ssh_base_cmd(self, host: str) -> list[str]:
        cmd = [
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "IdentitiesOnly=yes",
            "-o",
            "IdentityAgent=none",
            "-o",
            "BatchMode=yes",
            "-o",
            "PreferredAuthentications=publickey",
        ]
        if not self.config.key_path:
            raise RuntimeError("No SSH key found. Set FEATHER_SSH_KEY to a readable private key path.")
        cmd += ["-i", self.config.key_path, f"{self.config.ssh_user}@{host}"]
        return cmd

    @staticmethod
    def _remote_shell_path(path: str) -> str:
        # Preserve remote $HOME expansion while keeping spaces safe.
        if path == "~":
            return '"$HOME"'
        if path.startswith("~/"):
            return '"$HOME/' + path[2:] + '"'
        return shlex.quote(path)

    def remote_home(self, host: str | None = None) -> str:
        host = host or self.config.head_host
        if host not in self._remote_home_cache:
            home = subprocess.check_output([*self._ssh_base_cmd(host), 'printf %s "$HOME"'], text=True).strip()
            if not home:
                raise RuntimeError(f"Could not resolve remote home for {host}")
            self._remote_home_cache[host] = home
        return self._remote_home_cache[host]

    def remote_runtime_path(self, path: str, host: str | None = None) -> str:
        # Convert ~/... to absolute path for worker-side Python file APIs.
        host = host or self.config.head_host
        if path == "~":
            return self.remote_home(host)
        if path.startswith("~/"):
            return f"{self.remote_home(host)}/{path[2:]}"
        return path

    def ssh_lines(self, host: str, remote_cmd: str) -> list[str]:
        out = subprocess.check_output([*self._ssh_base_cmd(host), remote_cmd], text=True)
        return [line.strip() for line in out.splitlines() if line.strip()]

    def list_remote_images(self, input_host: str | None = None) -> list[str]:
        input_host = input_host or self.config.head_host
        d = self._remote_shell_path(self.config.remote_input_dir)
        cmd = f"find {d} -maxdepth 1 -type f \\( -iname '*.jpg' -o -iname '*.jpeg' \\) | sort"
        return self.ssh_lines(input_host, cmd)

    def list_remote_outputs(self, host: str) -> list[str]:
        d = self._remote_shell_path(self.config.remote_output_dir)
        cmd = f"find {d} -maxdepth 1 -type f -iname '*.jpg' | sort"
        return self.ssh_lines(host, cmd)

    def fetch_remote_file(self, host: str, remote_path: str, local_dir: Path) -> Path:
        local_dir.mkdir(parents=True, exist_ok=True)
        local_file = local_dir / Path(remote_path).name
        remote_cmd = f"cat {shlex.quote(remote_path)}"
        with open(local_file, "wb") as fh:
            subprocess.check_call([*self._ssh_base_cmd(host), remote_cmd], stdout=fh)
        return local_file
