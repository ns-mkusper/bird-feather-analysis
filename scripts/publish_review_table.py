"""Per-feather review table for a label run + segmentation folder, copied to Drive.

Builds the per-feather output table (one row per feather crop: feather ID, the
slide's label fields, judge score and flags, duplicates) with empty
`your_verdict` / `your_notes` columns, plus the slide table, the crops, the
judge panels and the overlays, and copies them to
gdrive:Feather_Molt_Project/label_runs/<run>/review/. Re-run to refresh: judge
scores still being computed show as 'pending'. --sheet also makes a Google Sheet
copy for verdicts, once (never overwritten).

    python scripts/publish_review_table.py data/label_runs/sample30-v4 data/sample_runs/test-sample30-seg
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import feather_duplicates, feather_judge, label_extract as lx, prompt_lab as pl  # noqa: E402
from src.output_names import find_overlay  # noqa: E402

FIRST = ["file_id", "feather_id", "seq", "your_verdict", "your_notes", "judge_score", "needs_review", "judge_notes",
         "judge_complete", "judge_single_feather", "judge_fragment", "judge_missing_part",
         "judge_background_included", "judge_other_feather_included", "judge_status", "duplicate_of",
         "bird_id", "bird_id_status", "species", "species_status", "age", "age_status", "sex", "sex_status",
         "year", "month", "day", "date_status", "crop", "qc_image"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("seg_dir")
    ap.add_argument("--remote", default="gdrive:Feather_Molt_Project/label_runs")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--sheet", action="store_true",
                    help="also create a Google Sheet copy for verdicts (only if none exists, so edits are never overwritten)")
    args = ap.parse_args()
    run, seg = Path(args.run_dir), Path(args.seg_dir)
    names = [l.strip() for l in (run / "manifest.txt").read_text().splitlines() if l.strip()]
    latest = lx.load_previous(run / "labels.jsonl")
    results = [latest[n] for n in names if n in latest]
    lx.attach_feathers_from_crops(results, seg)
    feather_duplicates.apply_to_results(results, seg)
    feathers = pl.list_feathers(seg, names)
    scores = pl.cached_judge(pl.current_prompts()["feather_judge_prompt"], feathers)

    out = run / "review"
    for sub in ("crops", "judge_panels", "overlays"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    rows = []
    for r in results:
        for f in r.feathers:
            res = scores.get(f["crop"])
            if res:
                f.update({"judge_" + k: v for k, v in res.items() if k not in ("panel", "match", "raw")})
                if res.get("panel"):
                    panel = Path(res["panel"])
                    shutil.copy(panel, out / "judge_panels" / (Path(f["crop"]).stem + "_qc.jpg"))
                    f["qc_image"] = f"judge_panels/{Path(f['crop']).stem}_qc.jpg"
            else:
                f["judge_status"] = "pending"
            f["needs_review"] = feather_judge.needs_review(f) if res else ""
            shutil.copy(seg / f["crop"], out / "crops" / f["crop"])
            f["crop"] = f"crops/{f['crop']}"
        ov = find_overlay(seg, r.file_id)
        if ov:
            shutil.copy(ov, out / "overlays" / ov.name)
        rows += lx.feather_table_rows(r, run.name)
    df = pd.DataFrame(rows)
    df["your_verdict"], df["your_notes"] = "", ""
    df = df[[c for c in FIRST if c in df] + [c for c in df if c not in FIRST]]
    df.to_csv(out / "feathers_review.csv", index=False)
    slides = pd.DataFrame([r.row() for r in results])
    slides.to_csv(out / "slides.csv", index=False)
    scored = df.judge_status.eq("ok").sum()
    print(f"{len(df)} feathers ({scored} scored, {df.judge_status.eq('pending').sum()} pending), "
          f"{len(slides)} slides -> {out}")
    if not args.no_upload:
        dest = f"{args.remote}/{run.name}/review"
        subprocess.run(["rclone", "copy", str(out), dest, "-q"], check=True)
        print("copied to", dest)
        if args.sheet:
            have = subprocess.run(["rclone", "lsf", dest, "--files-only", "-q"], capture_output=True, text=True).stdout
            if "feathers_review_sheet" not in have:
                sheet_csv = out.parent / "feathers_review_sheet.csv"
                shutil.copy(out / "feathers_review.csv", sheet_csv)
                subprocess.run(["rclone", "copy", str(sheet_csv), dest, "--drive-import-formats", "csv", "-q"],
                               check=True)
                sheet_csv.unlink()
                print("created Google Sheet feathers_review_sheet in", dest)
            else:
                print("Google Sheet already there; left untouched (your edits are kept)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
