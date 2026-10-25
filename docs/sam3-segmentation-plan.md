# Improving segmentation with SAM 3

The pipeline uses SAM 3 only through its text prompt (`"bird feather"`, plus two darker-feather prompts on
retry) on the whole slide at SAM 3's native 1008 px. SAM 3 can do more than that. This page lists what to try,
how it is measured, and what was found.

## Baseline: the 100-slide run (`slice100-20261019`)

| | |
|---|---|
| Feathers | 339 |
| Feathers that failed a check | 57 |
| Feather count right (30 hand-checked slides) | 29 / 30 |
| Missed-feather flags | 3 slides |

The failures fall into a few kinds:
- **Merged:** one mask covers several feathers, e.g. `BB17692 1998-im0513` B holds three feathers.
- **Split:** one feather becomes two masks, e.g. `A3818 2002-im1708`, a feather partly under the board.
- **Outline cut:** the mask stops short of the fringe or the downy base, e.g. `A2328 2000` C at the bottom.
- **Small measured holes:** pockets between separated barbs. These look like false alarms of the hole
  measurement (threshold 0.2% of the feather), not segmentation errors.

## Ideas, in the order to test them

1. **Example-box prompts.** Besides text, SAM 3 takes boxes as examples: positive ("find feathers like
   this one") and negative ("not this"). Give it a well-segmented feather from the same slide as a positive
   example, and the label and date areas as negatives. Aimed at merged and split feathers.
2. **A second pass per feather.** Re-segment each feather on a crop around it, with SAM 3's interactive
   predictor: a box around the feather, a positive point on it and negative points on its neighbours. The crop
   is far smaller than the slide, so SAM 3 sees it at a much higher effective resolution. Aimed at cut fringe and
   downy bases, and at separating touching feathers.
3. **Part prompts.** Text prompts for parts ("downy feather base", "feather fringe") to add back where a
   feather's check reports a side cut.
4. **Repair only flagged feathers.** Re-prompt only feathers that failed a check, instead of re-segmenting
   whole slides on retry.

Also to tune separately: the measured-hole threshold (0.2% → about 1%, or only when the model also sees a hole).

## How it is measured

Each variant goes through the pipeline's own processing (mask clean-up, crops, per-feather checks with the
same prompt and model, missed-feather check) on the same slides: the problem slides above plus clean slides as
controls. Compared per variant: feathers that fail a check and which check, feather counts against the hand
counts, and the overlays side by side. A variant is worth keeping if it fixes failures without breaking the
clean slides.

Script: `scripts/sam3_experiments.py` (runs on one worker node, outside the job queue).

## Results: ideas 1 and 2 (2026-10-19)

17 slides: 12 problem slides from the 100-slide run, 2 with hole flags only, 3 clean. Each went through the
pipeline 4 times. The run ID was `test-sam3exp-10190455`, on one worker node.

| Variant | Feathers failing a check, holes aside | Hole flags | Feather counts right | Mean score | s / slide |
|---|---|---|---|---|---|
| Text prompt (now) | 20 | 15 | 8/9 | 8.31 | 90 |
| + example boxes (idea 1) | 16 | 15 | 9/9 | 8.51 | 91 |
| + second pass per feather (idea 2) | 21 | 23 | 9/9 | 7.91 | 95 |
| both | 19 | 27 | 9/9 | 7.62 | 96 |

**Example boxes: a small, real gain. Switched on.**
- It fixes a split feather: `A3818 2002-im1708` goes from 5 crops to 4, matching the hand count, and the
  fragment is gone.
- One mask improved (`BX40715 2005` A grew 6.5%, and its outline-cut flag cleared).
- Clean slides are unchanged: mask areas move by at most 1.6%.
- It costs about 1 s per slide.
- It does not fix merged feathers: the `BB17692 1998` slides are unchanged.

**Second pass per feather: worse. Off.** The refined masks have more holes (23 and 27 hole flags against
15), and the scores drop.

**Checks vary between runs.** Four of the five flag changes between the text prompt and the example boxes
happened on masks whose area changed by less than 1%. Three flags cleared and one appeared on a clean slide.
So the per-feather checks are not repeatable to within a few feathers. Compare variants on mask changes and
feather counts, not only on flag counts.

### 100-slide rerun with example boxes (`slice100-sam3ex-20261019`)

Same 100 slides as `slice100-20261019`, now with example boxes.

| | Before | With example boxes |
|---|---|---|
| Feather count right (30 hand-checked slides) | 29/30 | 30/30 |
| Feathers | 339 | 338 |
| Feathers flagged (any reason) | 53 | 62 |
| Feathers flagged beyond holes | 28 | 32 |
| Duplicates marked | 4 | 3 |
| Label fields right (vs hand-checked) | 137/138 | 137/138 |

- **Masks barely change:** 320 of 334 feathers keep their area within 5%. The only count change is the fixed
  split (`A3818`).
- **The higher flag count is the checks varying:** 19 flags changed on masks within 1%.
- **The 14 changed masks mostly grew by 5-7%,** covering more fringe:
  - one outline-cut flag cleared;
  - two feathers picked up a measured-hole flag (the hole threshold again);
  - nothing else changed.
- **Duplicates:** the missing one is the match between a single feather and the crop that merges three
  feathers (`BB17692 im0513` B), which was never a clean duplicate.

So example boxes are a small, real gain: they fix splits and change nothing else for the worse.

Next:
- Merged feathers still need a fix: part prompts (idea 3) or flagged-only repair (idea 4).
- The second pass could be retried with the first mask as a mask prompt instead of box + points.
- Make the per-feather checks steadier: ask twice and flag on disagreement, or only on the checks that vary.
- Raise the measured-hole threshold (0.2% → about 1%). Hole flags are most of the flags (44 of 62).

## Follow-ups (2026-10-19)

1. **Measured holes.** Holes found only by measurement were 0.2-1.3% of the feather. The ones checked by eye
   were natural vane splits with background showing through. They now fail the check only from 2%
   (`feather_judge.fail_hole_frac`); smaller ones are recorded. A mask in pieces, and the model's own "holes"
   answer, still fail it.
2. **Steadier checks.** On 243 feathers whose masks did not change between the runs, 227 got identical
   answers on all nine checks. The ones that changed were mostly holes, other material inside and a cut at
   the bottom. A failed check now gets a second look on the same panel at 80% (`feather_judge.second_look`):
   - failed both times: a failure (`needs_review`, score, retry);
   - failed once: listed in `review_uncertain`.

   On the 24 feathers flagged in either run, the runs agree on the confirmed failures for 19, against 8 with
   one look. Requiring confirmation alone would have cleared 6 of 12 failures both runs had seen, so those
   stay visible as uncertain.
3. **Merged feathers.** SAM 3 does not merge them: on `BB17692 1998-im0513` it returns 8 separate masks of
   normal size. The three-feather crop came from the mask clean-up: "aligned pieces" (rule 3) joined parallel
   feathers lying partly over each other. Rule 3 now needs the pieces end to end, overlapping along the axis
   by at most a quarter of the shorter piece. The slide goes from 4 crops (three feathers fused) to 7: five
   whole feathers and two pieces of one mostly hidden under the board. Other slides are unchanged. It was the
   only rule-3 merge in both runs. A repair step for oversized SAM 3 masks was tried and removed, because
   there were none to repair.

## Repairs after the checks (2026-10-19)

- **Fragments are set aside.** A piece the clean-up marks small or narrow, which the check also calls a
  fragment or not a single feather on both looks, gets no letter or feather ID. It is kept as
  `<prefix>_Fragment_<n>.jpg` and a review row; the slide is retried like any failed check. A small piece
  the model sees as a whole feather stays a feather, flagged `small/narrow mask`.
- **Usable feathers.** `feathers.csv` marks each row `usable` or not with the reason; `feathers_usable.csv`
  holds only the usable ones.
- **Re-segmenting oversized masks on a crop: not needed.** SAM 3 returned no merged masks in either run;
  the one merged crop came from the clean-up (rule 3, fixed above).

## Decided: vane only, the quill (calamus) is cut (2026-10-25)

The analysis aims at the vane, so the calamus can be cropped out (supervisor's input, 2026-10-25): `crop.cleanup.calamus` cuts the bare quill below the vane on every feather. The question as it was recorded:

### Open question: keep the quill (stem) or cut to the full vane? (2026-10-20)

**Current behaviour (`crop.cleanup`):** on the white card, each feather is cut where its vane narrows to the quill.
That removes the glue dots, glue specks and the see-through downy base, but also the quill.

**What prompted the question.** On `AX87168 1990-im2328`, the original April run (`20260426-075440`: detector boxes,
then SAM 2) followed each quill cleanly through the glue dot and left the glue out. SAM 3's masks include the glue
dots; the crop only looks clean because the clean-up cuts the base off. So April kept a clean stem and the current
pipeline does not.

**To decide (with the supervisor):** does the analysis need the quill, e.g. for feather length?
- **Cut to the full vane** (now): crops without glue or stem.
- **Keep the stem, cut only the glue:** re-segment the base of feathers with glue, with SAM 3's point prompts
  (positive points along the quill, negative points on the glue patch found by `src/mask_cleanup.py`), aiming for
  April-style stems. Not built yet: about an hour of testing on the 16 glue feathers of the hand verdicts, checked
  against April's masks.

