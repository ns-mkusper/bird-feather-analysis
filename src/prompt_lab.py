"""Helpers for notebooks/prompt_lab.ipynb: try segmentation and scoring prompts.

Judge prompts run from here through the VLM server on any segmentation output
folder (crops from older worker code are located on their slide to rebuild the
mask). Results are cached per (prompt, crop), so re-running a cell is instant.
SAM 3 prompts/settings run on the workers through a sample run with
options={'segmentation': ...}; that needs the workers on this branch's code.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import pandas as pd

from src import feather_judge
from src import label_extract as lx
from src.crop_locate import crop_mask
from src.feather_processing import SEG_CONFIG, load_seg_config
from src.output_names import find_crops, find_overlay, seq_letter

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "data" / "prompt_lab"


def current_prompts() -> dict:
    cfg = load_seg_config()
    return {"sam3_prompts": cfg["sam3"]["prompts"], "sam3_retry_prompts": cfg["sam3"].get("retry_prompts", []),
            "sam3_conf": cfg["sam3"]["conf"], "postprocess": cfg["postprocess"],
            "feather_judge_prompt": cfg["feather_judge"]["prompt"], "slide_judge_prompt": cfg["slide_judge"]["prompt"]}


def list_feathers(seg_dir: str | Path, slides: list[str] | None = None, limit: int | None = None) -> list[dict]:
    """Feather crops in a segmentation folder, slide by slide, left to right."""
    seg_dir = Path(seg_dir)
    if slides is None:
        res = seg_dir / "task_results.json"
        slides = sorted(json.loads(res.read_text())) if res.exists() else sorted(
            {p.name.split("_")[0] + ".jpg" for p in seg_dir.glob("*_BoundingBoxes.jpg")})
    out = []
    for s in slides:
        for i, c in enumerate(find_crops(seg_dir, s)):
            out.append({"file_id": s, "seq": seq_letter(i), "crop": c.name, "crop_path": str(c)})
    return out[:limit] if limit else out


def _key(prompt: str, crop_path: str, extra: str = "") -> str:
    return hashlib.sha1(f"{prompt}\0{Path(crop_path).name}\0{extra}".encode()).hexdigest()[:16]


def run_judge(prompt: str, feathers: list[dict], slides_dir: str | Path = REPO / "data" / "raw_cache",
              judge_overrides: dict | None = None, log=print) -> pd.DataFrame:
    """Scores each feather with `prompt` (cached). Returns one row per feather."""
    jcfg = {**load_seg_config()["feather_judge"], **(judge_overrides or {}), "prompt": prompt}
    model = lx.load_config()["vlm"]["model"]
    client = lx.VLMClient(lx.load_config()["vlm"])
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "panels").mkdir(exist_ok=True)
    rows, slides = [], {}
    for k, f in enumerate(feathers):
        key = _key(prompt, f["crop_path"], model + json.dumps(judge_overrides or {}, sort_keys=True))
        cached = CACHE / f"{key}.json"
        if cached.exists():
            res = json.loads(cached.read_text())
        else:
            if f["file_id"] not in slides:
                slides[f["file_id"]] = cv2.imread(str(Path(slides_dir) / f["file_id"]))
            slide = slides[f["file_id"]]
            crop = cv2.imread(f["crop_path"])
            if slide is None or crop is None:
                res = {"status": "missing_image"}
            else:
                mask, match = crop_mask(slide, crop)
                res, panel = feather_judge.judge(client, slide, mask, crop, jcfg)
                panel_path = CACHE / "panels" / f"{Path(f['crop_path']).stem}_{key}.jpg"
                panel.save(panel_path, quality=85)
                res["panel"] = str(panel_path)
                res["match"] = round(match, 3)
            cached.write_text(json.dumps(res))
            log(f"  {k + 1}/{len(feathers)} {f['file_id']} {f['seq']}: {res.get('score')} {res.get('notes', '')}")
        rows.append({**{k2: f[k2] for k2 in ("file_id", "seq", "crop")}, **res})
    df = pd.DataFrame(rows)
    df["review"] = [feather_judge.needs_review({"judge_" + k: v for k, v in r.items()}) for r in rows]
    return df


def cached_judge(prompt: str, feathers: list[dict], judge_overrides: dict | None = None) -> dict[str, dict]:
    """Judge results already in the cache for `prompt`, by crop name (no model calls)."""
    model = lx.load_config()["vlm"]["model"]
    out = {}
    for f in feathers:
        p = CACHE / f"{_key(prompt, f['crop_path'], model + json.dumps(judge_overrides or {}, sort_keys=True))}.json"
        if p.exists():
            out[f["crop"]] = json.loads(p.read_text())
    return out


def compare(a: pd.DataFrame, b: pd.DataFrame, verdicts: str | Path | None = None,
            good_from: float = 7) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Side-by-side A/B table; with your verdicts CSV (crop, verdict = good|bad),
    also how often each prompt agrees with you (score >= good_from = good)."""
    cols = ["score", "complete", "fragment", "single_feather", "missing_part", "notes", "status"]
    m = a[["file_id", "seq", "crop"] + [c for c in cols if c in a]].merge(
        b[["crop"] + [c for c in cols if c in b]], on="crop", suffixes=("_A", "_B"))
    m["delta"] = m["score_B"] - m["score_A"]
    stats = None
    if verdicts and Path(verdicts).exists():
        v = pd.read_csv(verdicts).dropna(subset=["verdict"])
        v = v[v.verdict.isin(["good", "bad"])]
        m = m.merge(v[["crop", "verdict"]], on="crop", how="left")
        rows = []
        for side in ("A", "B"):
            j = m.dropna(subset=["verdict", f"score_{side}"])
            pred = j[f"score_{side}"].ge(good_from).map({True: "good", False: "bad"})
            rows.append({"prompt": side, "feathers with your verdict": len(j),
                         "agree": int((pred == j.verdict).sum()),
                         "bad called good": int(((pred == "good") & (j.verdict == "bad")).sum()),
                         "good called bad": int(((pred == "bad") & (j.verdict == "good")).sum())})
        stats = pd.DataFrame(rows)
    return m, stats


def write_verdict_template(feathers: list[dict], path: str | Path) -> Path:
    """A CSV to fill in by hand: verdict = good | bad for each crop (keeps existing calls)."""
    path = Path(path)
    old = {}
    if path.exists():
        old = {r["crop"]: r for r in csv.DictReader(open(path, newline=""))}
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "seq", "crop", "verdict", "why"])
        w.writeheader()
        for f in feathers:
            o = old.get(f["crop"], {})
            w.writerow({"file_id": f["file_id"], "seq": f["seq"], "crop": f["crop"],
                        "verdict": o.get("verdict", ""), "why": o.get("why", "")})
    return path


def save_prompt(kind: str, prompt: str) -> None:
    """Writes a prompt into configs/segmentation.json (kind: feather_judge | slide_judge)."""
    cfg = json.loads(Path(SEG_CONFIG).read_text())
    cfg[kind]["prompt"] = prompt
    Path(SEG_CONFIG).write_text(json.dumps(cfg, indent=2) + "\n")


def workers_have_options_support(timeout: float = 5) -> tuple[bool, dict]:
    """True when the cluster workers run this branch's code (the label task is registered)."""
    from src.notebook_remote_runtime import create_celery_client, load_notebook_config_from_env

    app = create_celery_client(load_notebook_config_from_env())
    reg = app.control.inspect(timeout=timeout).registered() or {}
    ok = bool(reg) and all("src.celery_tasks.extract_labels" in tasks for tasks in reg.values())
    return ok, {w: sorted(t) for w, t in reg.items()}


def run_seg_experiment(slides: list[str], options: dict | None, tag: str, timeout: int = 5400) -> Path:
    """Segments `slides` on the workers with segmentation overrides; returns the
    local output folder (cached: an existing folder with results is reused)."""
    from src import sample_run

    h = hashlib.sha1(json.dumps([sorted(slides), options], sort_keys=True).encode()).hexdigest()[:8]
    run_id = f"test-promptlab-{tag}-{h}"
    out = REPO / "data" / "sample_runs" / run_id
    if (out / "task_results.json").exists():
        return out
    return sample_run.run(match=slides, timeout=timeout, run_id=run_id,
                          options={"segmentation": options} if options else None)


def seg_summary(seg_dir: str | Path, gold: str | Path | None = REPO / "data" / "label_gold.csv") -> pd.DataFrame:
    """Per slide: feathers kept, hand count, clean-up actions, judge scores."""
    seg_dir = Path(seg_dir)
    res = json.loads((seg_dir / "task_results.json").read_text()) if (seg_dir / "task_results.json").exists() else {}
    g = lx.load_gold(gold) if gold and Path(gold).exists() else {}
    rows = []
    for s in sorted(res or {p.name.split("_")[0] + ".jpg" for p in seg_dir.glob("*_BoundingBoxes.jpg")}):
        recs = (res.get(s) or {}).get("feather_ids") or []
        scores = [r.get("judge_score") for r in recs if isinstance(r.get("judge_score"), (int, float))]
        rows.append({"file_id": s, "feathers": len(find_crops(seg_dir, s)),
                     "hand_count": (g.get(s) or {}).get("feather_count") or None,
                     "merged_or_dropped": sum(bool(r.get("postprocess")) for r in recs),
                     "min_score": min(scores) if scores else None,
                     "mean_score": round(sum(scores) / len(scores), 2) if scores else None,
                     "needs_review": sum(bool(r.get("needs_review")) for r in recs),
                     "overlay": str(find_overlay(seg_dir, s) or "")})
    df = pd.DataFrame(rows)
    if len(df) and df.hand_count.notna().any():
        df["count_ok"] = df.feathers == pd.to_numeric(df.hand_count, errors="coerce")
    return df
