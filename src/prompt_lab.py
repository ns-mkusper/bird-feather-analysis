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
            "feather_judge_prompt": cfg["feather_judge"]["prompt"],
            "feather_score": cfg["feather_judge"].get("score", feather_judge.DEFAULT_SCORE),
            "inventory_prompt": cfg["inventory"]["prompt"]}


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


# Per-check calibration: one column per check in the verdicts CSV; "cut"/"complete"
# for the sides, true/false for the rest, blank = not checked.
CHECKS = ["outline_left", "outline_right", "outline_top", "outline_bottom", "holes_or_gaps", "fragment",
          "single_feather", "other_feather_inside", "other_material_inside"]
# The answer that means "this check found a problem"
_FLAG = {**{c: "cut" for c in CHECKS[:4]}, "holes_or_gaps": True, "fragment": True, "single_feather": False,
         "other_feather_inside": True, "other_material_inside": True}


def _norm(check: str, v):
    """CSV/JSON value -> 'cut'|'complete' or True|False; None when blank."""
    if v is None or (isinstance(v, float) and pd.isna(v)) or str(v).strip() == "":
        return None
    if check.startswith("outline_"):
        t = str(v).strip().lower()
        return t if t in ("cut", "complete") else None
    t = str(v).strip().lower()
    return True if t in ("true", "1", "yes", "y") else False if t in ("false", "0", "no", "n") else None


def _model_flags(row) -> dict:
    """Model answer per check; holes_or_gaps includes the measured holes/split mask."""
    return {c: _norm(c, row.get(c)) for c in CHECKS}


def _good(df: pd.DataFrame) -> pd.Series:
    """good = the feather passed every check (the pipeline's decision), bad = any check failed."""
    return pd.Series([("good" if feather_judge.needs_review({"judge_" + k: v for k, v in r.items()}) == ""
                       else "bad") if r.get("status") == "ok" else None
                      for r in df.to_dict("records")], index=df.index)


def compare(a: pd.DataFrame, b: pd.DataFrame, verdicts: str | Path | None = None
            ) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Side-by-side A/B table; with your verdicts CSV (crop, verdict = good|bad),
    also how often each prompt's pass/fail (all checks passed = good) agrees
    with you. The derived score is shown for sorting only."""
    cols = ["score", "sides_cut", "holes_or_gaps", "holes", "mask_parts", "fragment", "single_feather",
            "other_feather_inside", "other_material_inside", "model_score", "notes", "status"]
    a = a.assign(decision=_good(a))
    b = b.assign(decision=_good(b))
    m = a[["file_id", "seq", "crop", "decision"] + [c for c in cols if c in a]].merge(
        b[["crop", "decision"] + [c for c in cols if c in b]], on="crop", suffixes=("_A", "_B"))
    m["delta"] = m["score_B"] - m["score_A"]
    stats = None
    if verdicts and Path(verdicts).exists():
        v = pd.read_csv(verdicts, dtype=str).fillna("")
        v = v[v.verdict.isin(["good", "bad"])]
        m = m.merge(v[["crop", "verdict"]], on="crop", how="left")
        rows = []
        for side in ("A", "B"):
            j = m.dropna(subset=["verdict", f"decision_{side}"])
            pred = j[f"decision_{side}"]
            rows.append({"prompt": side, "feathers with your verdict": len(j),
                         "agree": int((pred == j.verdict).sum()),
                         "bad called good": int(((pred == "good") & (j.verdict == "bad")).sum()),
                         "good called bad": int(((pred == "bad") & (j.verdict == "good")).sum())})
        stats = pd.DataFrame(rows)
    return m, stats


def check_agreement(results: pd.DataFrame, verdicts: str | Path) -> pd.DataFrame:
    """Per check: how often the model's answer matches your hand check, over the
    feathers where you filled that column. 'missed problem' = you flagged it and
    the model passed it (the costly kind); 'false alarm' = the reverse."""
    if not Path(verdicts).exists():
        return pd.DataFrame()
    v = pd.read_csv(verdicts, dtype=str).fillna("")
    v = {r["crop"]: r for r in v.to_dict("records")}
    rows = []
    for check in CHECKS:
        n = agree = missed = false_alarm = 0
        for r in results.to_dict("records"):
            if r.get("status") != "ok" or r.get("crop") not in v:
                continue
            want = _norm(check, v[r["crop"]].get(check))
            got = _model_flags(r)[check]
            if want is None or got is None:
                continue
            n += 1
            agree += want == got
            missed += (want == _FLAG[check]) and (got != _FLAG[check])
            false_alarm += (got == _FLAG[check]) and (want != _FLAG[check])
        rows.append({"check": check, "feathers you checked": n, "agree": agree,
                     "agreement": round(agree / n, 3) if n else None,
                     "missed problem": missed, "false alarm": false_alarm})
    return pd.DataFrame(rows)


def write_verdict_template(feathers: list[dict], path: str | Path) -> Path:
    """A CSV to fill in by hand: verdict = good | bad for each crop and, for the
    per-check calibration, each check (outline_* = cut|complete, the rest
    true|false; blank = not checked). Existing entries are kept."""
    path = Path(path)
    old = {}
    if path.exists():
        old = {r["crop"]: r for r in csv.DictReader(open(path, newline=""))}
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_id", "seq", "crop", "verdict", *CHECKS, "why"])
        w.writeheader()
        for f in feathers:
            o = old.get(f["crop"], {})
            w.writerow({"file_id": f["file_id"], "seq": f["seq"], "crop": f["crop"],
                        **{k: o.get(k, "") for k in ("verdict", *CHECKS, "why")}})
    return path


def save_prompt(kind: str, prompt: str) -> None:
    """Writes a prompt into configs/segmentation.json (kind: feather_judge | inventory)."""
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


def run_inventory(prompt: str, seg_dir: str | Path, slides: list[str], hide: int | None = 1,
                  slides_dir: str | Path = REPO / "data" / "raw_cache", log=print) -> pd.DataFrame:
    """Tests the missed-feather prompt: rebuilds each slide's masks from its crops
    and hides one (`hide` = 0-based position left to right; None = hide nothing),
    so the right answer is known. Cached per (prompt, slide, hide)."""
    from src import feather_inventory

    icfg = {**load_seg_config()["inventory"], "prompt": prompt}
    client = lx.VLMClient(lx.load_config()["vlm"])
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "inventory").mkdir(exist_ok=True)
    rows = []
    for s in slides:
        key = _key(prompt, s, f"inventory hide={hide}")
        cached = CACHE / f"inv_{key}.json"
        if cached.exists():
            rows.append(json.loads(cached.read_text()))
            continue
        slide = cv2.imread(str(Path(slides_dir) / s))
        crops = find_crops(seg_dir, s)
        if slide is None or not crops:
            continue
        masks = [crop_mask(slide, cv2.imread(str(c)))[0] for c in crops]
        masks.sort(key=lambda m: m.nonzero()[1].mean())
        expected = None
        if hide is not None and len(masks) > hide:
            masks = masks[:hide] + masks[hide + 1:]
            expected = seq_letter(hide)
        items, summary, overlay = feather_inventory.check(client, slide, masks, icfg)
        img_path = CACHE / "inventory" / f"{Path(s).stem}_{key}.jpg"
        overlay.save(img_path, quality=85)
        found = [seq_letter(i) for i, it in enumerate(items) if it["kind"] == "missed"]
        row = {"file_id": s, "segmented_shown": len(masks), "hidden": expected or "", "found_missed": ",".join(found),
               "correct": (found == [expected]) if expected else (found == []),
               **{k: summary[k] for k in ("inventory_status", "inventory_total", "inventory_note")},
               "overlay": str(img_path)}
        cached.write_text(json.dumps(row))
        log(f"  {s}: hidden {expected} found {found} ({summary['inventory_status']})")
        rows.append(row)
    return pd.DataFrame(rows)
