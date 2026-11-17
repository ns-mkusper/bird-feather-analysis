"""Re-derive a finished run's label tables under the current rules (label_extract.reclassify,
src/card_rules.py, src/slide_annotations.py) from its labels.jsonl, into a NEW folder. Feathers,
masks and crops are not touched (scripts/rebuild_run.py rebuilds those).

    python scripts/relabel_run.py data/label_runs/full-20261114 --to data/label_runs/full-20261114-cards \\
        --handwriting data/review/label_handwriting_slice100.csv

--complete also asks the model for what the saved readings lack (fields added to the config since,
the age box question for slides whose age box was not read) and takes a second look at fields left
partly visible or needing review (label_extract.second_look), one slide at a time, from the slides
in --slides (default data/raw_cache) and the VLM server (FEATHER_VLM_ENDPOINTS). Only those
questions are asked; the rest of each reading is kept.
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from src import label_extract as lx  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="folder with the run's labels.jsonl")
    ap.add_argument("--to", required=True, help="new folder for the tables (never overwritten)")
    ap.add_argument("--handwriting", help="CSV: file_id, handwritten, fields, note, source")
    ap.add_argument("--gold", default=str(REPO / "data/label_gold.csv"))
    ap.add_argument("--complete", action="store_true", help="ask the model for what the saved readings lack")
    ap.add_argument("--slides", default=str(REPO / "data/raw_cache"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--field-age-without-box-age", action="store_true",
                    help="read field_age (the plumage age in the species row) only on slides with no secured box "
                         "age (RC, empty box, covered); no other model questions")
    ap.add_argument("--only-problems", action="store_true",
                    help="with --complete: only slides with label problems under the current rules (plus --gold slides), "
                         "and no new fields (e.g. field_age) for the rest of the collection")
    args = ap.parse_args()

    src, out = Path(args.run), Path(args.to)
    if out.exists():
        sys.exit(f"{out} exists: choose another --to (this script never overwrites)")
    cfg = lx.load_config()
    results = lx.load_previous(src / "labels.jsonl")
    if not results:
        sys.exit(f"no readings in {src / 'labels.jsonl'}")
    out.mkdir(parents=True)

    if args.field_age_without_box_age:
        import copy

        from src import card_rules, slide_annotations
        trial = {k: copy.deepcopy(r) for k, r in results.items()}
        for r in trial.values():
            lx.reclassify(r, cfg)
            card_rules.apply(r, cfg)
        slide_annotations.annotate(list(trial.values()), list(trial))
        has_age = (lx.READABLE, card_rules.FROM_COLOUR, "inferred")
        todo = [r for k, r in results.items() if "field_age" not in r.fields and not r.has_model_error
                and (trial[k].fields.get("age") is None or trial[k].fields["age"].status not in has_age)]
        print(f"--field-age-without-box-age: {len(todo)} slides")
        client = lx.VLMClient(cfg["vlm"])

        def read_field_age(r):
            img = Path(args.slides) / r.file_id
            if not img.exists():
                return "no slide"
            image = lx.Image.open(img)
            image.load()
            r.fields["field_age"] = lx.read_field(client, image, "field_age", cfg["fields"]["field_age"], cfg)
            return r.fields["field_age"].status

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            print({k: v for k, v in __import__("collections").Counter(pool.map(read_field_age, todo)).items()})

    if args.complete:
        client = lx.VLMClient(cfg["vlm"])
        todo = [r for r in results.values() if not r.has_model_error]
        if args.only_problems:
            import copy

            from src import card_rules, slide_annotations
            trial = {k: copy.deepcopy(r) for k, r in results.items()}
            for r in trial.values():
                lx.reclassify(r, cfg)
                card_rules.apply(r, cfg)
            slide_annotations.annotate(list(trial.values()), list(trial),
                                       slide_annotations.load_handwriting(args.handwriting))
            gold = set(lx.load_gold(args.gold)) if Path(args.gold).exists() else set()
            todo = [r for r in todo if trial[r.file_id].extra.get("label_problems") or r.file_id in gold]
            print(f"--only-problems: {len(todo)} slides")
        # birds whose slides disagree on sex: read sex again with the case-keeping prompt, so a
        # supposed (small-letter) sex is told apart from a certain one (src/card_rules.py)
        sexes: dict[str, set] = {}
        for r in todo:
            b, s = r.fields.get("bird_id"), r.fields.get("sex")
            if b is not None and b.value and s is not None and s.status == lx.READABLE:
                sexes.setdefault(b.value, set()).add(s.value)
        sex_again = {r.file_id for r in todo if (b := r.fields.get("bird_id")) is not None
                     and len(sexes.get(b.value, ())) > 1}

        def one(r):
            img = Path(args.slides) / r.file_id
            if not img.exists():
                return r.file_id, "no slide"
            lx.reclassify(r, cfg)       # the age box question depends on the current age status
            asked = lx.complete_reading(r, img, client, cfg, new_fields=not args.only_problems)
            if r.file_id in sex_again:
                image = lx.Image.open(img)
                image.load()
                r.fields["sex"] = lx.read_field(client, image, "sex", cfg["fields"]["sex"], cfg)
                asked = True
            from src import card_rules
            card_rules.apply(r, cfg)
            looks = [n for n in (cfg.get("card") or {}).get("second_look", {}).get("fields", [])
                     if (f := r.fields.get(n)) is not None and f.status in ("partly_visible", "needs_review")]
            if looks:
                image = lx.Image.open(img)
                image.load()
                for n in looks:
                    lx.second_look(r, n, client, image, cfg)
                asked = True
            return r.file_id, "asked" if asked else "complete"

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            done = list(pool.map(one, todo))
        counts = {k: sum(v == k for _, v in done) for k in ("asked", "complete", "no slide")}
        print(f"--complete: {counts}")

    with open(out / "labels.jsonl", "w") as fh:
        for r in results.values():
            fh.write(json.dumps(r.to_json()) + "\n")
    names = list(results)
    summary = lx.finish_run(out, names, cfg, args.gold if Path(args.gold).exists() else None,
                            siblings=names, handwriting=args.handwriting)
    print(json.dumps({k: summary.get(k) for k in ("slides", "needs_review", "feathers", "feathers_usable",
                                                  "slides_with_label_problems", "gold")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
