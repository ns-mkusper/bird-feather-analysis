"""Re-run the per-feather checks of a finished run with other settings or another model, on
exactly the run's masks (feather_masks/, saved by the pipeline), and score them against your
hand verdicts (notebooks/feather_verdicts.ipynb).

    python scripts/rejudge.py slice100-masks-20261019 --variant saved \\
        --set '{"judge_mask": "saved", "prompt_add": "trim"}' --sample data/review/verdict_sample_<run>.csv
    python scripts/rejudge.py <run> --compare          # every variant run so far vs the verdicts

--set takes feather_judge overrides (judge_mask, enhance_contrast, panel_height, prompt, ...).
"prompt_add": names from PROMPT_ADD below, added to the configured prompt. --endpoint and --model
point at another VLM server (e.g. a second model on another node). Results: data/review/rejudge/<run>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import feather_judge as fj, label_extract as lx  # noqa: E402
from src.feather_processing import equalized, load_seg_config, make_feather  # noqa: E402

TRIM = ("The feather's base is trimmed off on purpose: the bottom of the outline may be a straight horizontal line "
        "across the feather. That straight cut is not a fault: answer \"complete\" for the bottom when the only cut "
        "there is that straight line, and ignore everything below it (base, glue, card).\n\n")
PROMPT_ADD = {"trim": ("1. Outline, side by side:", TRIM + "1. Outline, side by side:")}   # now in the shipped prompt


def judge_cfg(overrides: dict) -> dict:
    cfg = dict(load_seg_config()["feather_judge"])
    adds = overrides.pop("prompt_add", [])
    cfg.update(overrides)
    for name in ([adds] if isinstance(adds, str) else adds):
        old, new = PROMPT_ADD[name]
        assert old in cfg["prompt"], name
        cfg["prompt"] = cfg["prompt"].replace(old, new)
    return cfg


def rejudge(run: str, variant: str, overrides: dict, sample: Path | None, endpoint: str | None,
            model: str | None, cleanup: bool = False, slides_dir: Path | None = None) -> Path:
    seg = REPO / "data/sample_runs" / run
    fdf = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv")
    fdf = fdf[(fdf.segment_status == "segmented") & fdf.mask_image.notna()]
    if sample is not None:
        fdf = fdf[fdf.crop.isin(set(pd.read_csv(sample).crop))]
    cfg = judge_cfg(dict(overrides))
    seg_cfg = load_seg_config()
    trim = float(seg_cfg["crop"].get("trim_bottom_frac", 0.12))
    ccfg = {**(seg_cfg["crop"].get("cleanup") or {}), "enabled": True} if cleanup else None
    all_f = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv")
    med_area = all_f[all_f.segment_status == "segmented"].groupby("file_id").area_px.median().to_dict()
    vcfg = dict(lx.load_config()["vlm"])
    if model:
        vcfg["model"] = model
    if endpoint:
        vcfg["endpoints"] = [endpoint]
    client = lx.VLMClient(vcfg)
    if endpoint:
        client._endpoints = iter(lambda: endpoint, None)   # ignore FEATHER_VLM_ENDPOINTS
    out_dir = REPO / "data/review/rejudge" / run / variant
    (out_dir / "panels").mkdir(parents=True, exist_ok=True)
    (out_dir / "settings.json").write_text(json.dumps({"overrides": overrides, "endpoint": endpoint,
                                                       "model": vcfg["model"], "cleanup": cleanup}, indent=1))
    done = {}
    res_path = out_dir / "results.jsonl"
    if res_path.exists():
        done = {json.loads(l)["crop"]: l for l in res_path.read_text().splitlines() if l.strip()}
    slides: dict = {}
    with open(res_path, "a") as fh:
        for f in fdf.itertuples():
            if f.crop in done:
                continue
            if f.file_id not in slides:
                slides.clear()
                slides[f.file_id] = cv2.imread(str((slides_dir or REPO / "data/raw_cache") / f.file_id))
            img = slides[f.file_id]
            full = cv2.imread(str(seg / f.mask_image), 0) > 0
            before = make_feather(img, full, trim, 0)
            feather = make_feather(img, full, trim, 0, ccfg)
            removed = 1 - feather["mask"].sum() / max(1, before["mask"].sum())
            rel_area = feather["full_mask"].sum() / med_area[f.file_id] if med_area.get(f.file_id) else None
            res, panel = fj.judge(client, img, feather["mask"] if cfg.get("judge_mask") == "saved" else feather["full_mask"],
                                  equalized(feather["crop"]), cfg, rel_area)
            panel.save(out_dir / "panels" / (Path(f.crop).stem + ".jpg"), quality=85)
            rec = {"judge_" + k: v for k, v in res.items() if k != "raw"}
            row = {"crop": f.crop, "file_id": f.file_id, "status": res.get("status"), "score": res.get("score"),
                   "needs_review": fj.needs_review({**rec, "postprocess": f.postprocess}),
                   "uncertain": fj.uncertain_review(rec), "notes": res.get("notes"), "raw": res.get("raw"),
                   "cleanup": feather["record"]["mask_cleanup"], "removed": round(float(removed), 4),
                   "hidden": res.get("hidden"), "straight": res.get("straight_outline")}
            fh.write(json.dumps(row, default=str) + "\n"); fh.flush()
            print(variant, f.crop, row["score"], row["needs_review"] or "pass", flush=True)
    return res_path


def same_crop(a: Path, b: Path) -> bool:
    """The same feather crop in two runs (verdicts given on one run apply to the other)."""
    x, y = cv2.imread(str(a)), cv2.imread(str(b))
    return x is not None and y is not None and x.shape == y.shape and float(np.abs(x.astype(int) - y).mean()) < 2


def _rehide(run: str, x: dict) -> str:
    """needs_review under the current rules: measured 'hidden' from the raw segmentation, and model
    answers that need support (feather_judge.needs_support) dropped without it."""
    seg_cfg = load_seg_config()
    jcfg = {**seg_cfg["feather_judge"], "hidden": {**seg_cfg["feather_judge"].get("hidden", {}), "enabled": True}}
    fdf = _feathers(run)
    f = fdf.loc[x["crop"]]
    img = cv2.imread(str(REPO / "data/raw_cache" / f.file_id))
    raw = cv2.imread(str(REPO / "data/sample_runs" / run / f.mask_image), 0) > 0
    feather = make_feather(img, raw, float(seg_cfg["crop"].get("trim_bottom_frac", 0.12)), 0,
                           {**(seg_cfg["crop"].get("cleanup") or {}), "enabled": True})
    med = fdf[fdf.file_id == f.file_id].area_px.median()
    level = fj.hidden_level(fj.straight_outline(raw), raw.sum() / med if med else None, jcfg, img)
    m = fj.mask_metrics(feather["full_mask"], jcfg.get("min_hole_frac", 0.002))
    need = jcfg.get("needs_support", [])
    items = []
    for i in [i for i in x["needs_review"].split("; ") if i]:
        if i.endswith("hidden"):
            continue
        if i == "holes or gaps" and "holes_or_gaps" in need and (m["hole_frac"] or 0) < float(jcfg.get("fail_hole_frac", 0.02)) \
                and m["mask_parts"] <= 1:
            continue
        if i == "fragment" and "fragment" in need and "possible fragment" not in str(f.postprocess):
            continue
        if i == "not a single feather" and "single_feather" in need and m["mask_parts"] <= 1 \
                and "other feather inside" not in x["needs_review"] and "fragment" not in x["needs_review"] \
                and "possible fragment" not in str(f.postprocess):
            continue
        items.append(i)
    if level != "none":
        items.append(f"{'partly' if level == 'part' else 'mostly'} hidden")
    return "; ".join(items)


_FEATHERS: dict = {}


def _feathers(run: str) -> pd.DataFrame:
    if run not in _FEATHERS:
        d = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv")
        _FEATHERS[run] = d[d.segment_status == "segmented"].set_index("crop", drop=False)
    return _FEATHERS[run]


def same_mask(run_a: str, run_b: str, crop: str) -> bool:
    """The same raw segmentation of a feather in two runs (both saved feather_masks/)."""
    a = cv2.imread(str(REPO / "data/sample_runs" / run_a / "feather_masks" / (Path(crop).stem + ".png")), 0)
    b = cv2.imread(str(REPO / "data/sample_runs" / run_b / "feather_masks" / (Path(crop).stem + ".png")), 0)
    if a is None or b is None or a.shape != b.shape:
        return False
    a, b = a > 0, b > 0
    return (a & b).sum() / max(1, (a | b).sum()) > 0.98


def compare(run: str, verdicts: Path, via: str | None = None) -> pd.DataFrame:
    """Per variant, on the feathers you judged: bad feathers it flagged (caught), bad ones it passed
    (missed, the costly kind) and good ones it flagged (false alarms). Verdicts from another run
    (verdicts_<other run>.csv) count only where the crop is identical in both runs."""
    v = pd.read_csv(verdicts, dtype=str).fillna("")
    v = v[v.verdict.isin(["good", "bad"])].set_index("crop")
    other = verdicts.stem.removeprefix("verdicts_")
    if other != run:
        if via:   # crops changed (e.g. glue clean-up): link through a run with the same crops and saved masks
            keep = [c for c in v.index if same_crop(REPO / "data/sample_runs" / other / c, REPO / "data/sample_runs" / via / c)
                    and same_mask(via, run, c)]
            print(f"verdicts from {other}: {len(keep)} of {len(v)} feathers with the same segmentation in {run} (via {via})")
        else:
            keep = [c for c in v.index if same_crop(REPO / "data/sample_runs" / other / c, REPO / "data/sample_runs" / run / c)]
            print(f"verdicts from {other}: {len(keep)} of {len(v)} crops identical in {run}")
        v = v.loc[keep]
    rows = []
    variants = {"run (as shipped)": None, **{p.name: p for p in sorted((REPO / "data/review/rejudge" / run).glob("*"))
                                             if (p / "results.jsonl").exists()}}
    fdf = pd.read_csv(REPO / "data/label_runs" / run / "feathers.csv").fillna({"needs_review": ""})
    probs = [c for c in v.columns if c not in ("file_id", "feather_id", "verdict", "note", "source")]
    glue_only = {c for c in v.index if v.loc[c, "verdict"] == "bad"
                 and [p for p in probs if v.loc[c, p] == "x"] == ["glue or background inside"]}
    for name, p in variants.items():
        removed = {}
        if p is None:
            flags = dict(zip(fdf.crop, fdf.needs_review != ""))
            if "mask_cleanup" in fdf:   # the run's own glue/card clean-up (a cut, not a measured share)
                removed = {c: 0.01 for c, n in zip(fdf.crop, fdf.mask_cleanup.fillna("")) if n}
        else:
            r = [json.loads(l) for l in (p / "results.jsonl").read_text().splitlines() if l.strip()]
            if "hidden" in json.loads((p / "settings.json").read_text())["overrides"]:
                r = [{**x, "needs_review": _rehide(run, x)} for x in r]     # the current hidden rule
            flags = {x["crop"]: bool(x["needs_review"]) for x in r}
            removed = {x["crop"]: x.get("removed") or 0 for x in r}
        common = [c for c in v.index if c in flags]
        bad = [c for c in common if v.loc[c, "verdict"] == "bad"]
        good = [c for c in common if v.loc[c, "verdict"] == "good"]
        fixed = [c for c in bad if not flags[c] and c in glue_only and removed.get(c, 0) > 0]
        damaged = [c for c in good if removed.get(c, 0) > 0.10]   # the chosen cut to the full vane removes up to ~9%
        rows.append({"variant": name, "judged": len(common), "bad flagged": sum(flags[c] for c in bad),
                     "bad fixed (glue removed)": len(fixed),
                     "bad missed": sum(not flags[c] for c in bad) - len(fixed),
                     "good flagged (false alarm)": sum(flags[c] for c in good),
                     "good cut >10%": len(damaged),
                     "right": f"{len(bad) - (sum(not flags[c] for c in bad) - len(fixed)) + sum(not flags[c] and c not in damaged for c in good)}/{len(common)}"})
    return pd.DataFrame(rows).set_index("variant")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--variant")
    ap.add_argument("--set", default="{}", help="feather_judge overrides as JSON")
    ap.add_argument("--sample", help="CSV with a crop column: judge only these feathers")
    ap.add_argument("--endpoint")
    ap.add_argument("--model")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--cleanup", action="store_true", help="remove glue/card first (crop.cleanup)")
    ap.add_argument("--slides", help="folder with the slide images (default data/raw_cache)")
    ap.add_argument("--verdicts", help="verdicts CSV (default: data/review/verdicts_<run>.csv)")
    ap.add_argument("--via", help="with --compare: a run with the verdicts' crops and saved masks, to link them to this run")
    args = ap.parse_args()
    verdicts = Path(args.verdicts) if args.verdicts else REPO / "data/review" / f"verdicts_{args.run}.csv"
    if args.compare:
        print(compare(args.run, verdicts, args.via).to_string())
        return 0
    rejudge(args.run, args.variant, json.loads(args.set), Path(args.sample) if args.sample else None,
            args.endpoint, args.model, args.cleanup, Path(args.slides) if args.slides else None)
    return 0


if __name__ == "__main__":
    sys.exit(main())
