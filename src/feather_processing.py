import gc
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, field

# conda-forge's torchvision has no Metal kernel for nms (used by SAM 3 and the
# box NMS below); let those few ops run on the CPU instead of failing.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image
from ultralytics import SAM

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_VLM_MODEL = "mlx-community/Qwen3.8-27B-4bit"
SEG_CONFIG = os.path.join(REPO_ROOT, "configs", "segmentation.json")


def _deep_merge(base: dict, over: dict | None) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_seg_config(overrides: dict | None = None) -> dict:
    """configs/segmentation.json (re-read per slide, so edits apply without a
    restart), with per-task overrides on top."""
    with open(SEG_CONFIG) as fh:
        cfg = json.load(fh)
    return _deep_merge(cfg, overrides)


@dataclass
class ProcessResult:
    image_path: str
    success: bool
    reason: str = ""
    feathers_saved: int = 0
    vlm_score: float | None = None
    metadata_source: str = "unknown"
    bird_id: str = "UNKNOWN"
    date: str = "UNKNOWN"
    vlm_all_feathers_covered: bool | None = None
    vlm_background_leakage_detected: bool | None = None
    vlm_green_boxes_grouped_feathers: bool | None = None
    vlm_notes: str = ""
    processing_profile: str = "default"
    # One record per saved crop, left to right: seq (A, B, ...), crop file,
    # feather_id (<bird_id>_<period>_<seq>, when labels were read) with where
    # bird_id/period came from, mask measurements, clean-up notes and the
    # per-feather VLM judge result (judge_*). Fragments follow (segment_status
    # 'fragment', crop '<prefix>_Fragment_<n>.jpg', no letter or ID), then possibly
    # missed feathers as review rows (segment_status 'possible_missed', no letter,
    # crop or ID).
    feather_ids: list = field(default_factory=list)
    # Slide-level view of the per-feather checks (feather_judge.slide_checks);
    # the retry decision uses these, not a score.
    qa_checks: dict = field(default_factory=dict)


def make_feather(img: np.ndarray, full_mask: np.ndarray, trim: float, min_part: float,
                 cleanup: dict | None = None) -> dict | None:
    """The saved crop of one feather mask: the slide pixels inside it on black, the bottom
    `trim` share of its box cut off (the quill base and glue); None for an empty mask.
    With `cleanup` (crop.cleanup), glue and card found by src/mask_cleanup.py are removed
    first; the crop's bottom is then the higher of that cut and the trim line."""
    from src import segment_postprocess as sp

    notes: list[str] = []
    trim_ref = raw_mask = full_mask
    if cleanup and cleanup.get("enabled") and full_mask.any():
        from src import mask_cleanup

        full_mask, _, notes = mask_cleanup.clean(img, full_mask, cleanup)

    contours, _ = cv2.findContours(full_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # all parts of a merged feather (pieces across a board edge), not just the largest
    contours = [c for c in contours if cv2.contourArea(c) >= min_part] or \
        ([max(contours, key=cv2.contourArea)] if contours else [])
    if not contours:
        return None
    x, y, w, h = cv2.boundingRect(np.vstack(contours))
    ry, rh = cv2.boundingRect(trim_ref.astype(np.uint8))[1::2]           # trim line of the uncleaned mask
    y_max = min(y + h, ry + rh - int(rh * trim))
    final_mask = np.zeros_like(img)
    cv2.drawContours(final_mask, contours, -1, (255, 255, 255), -1)
    final_mask[y_max:, :] = 0
    feather_crop = cv2.bitwise_and(img, final_mask)[y:y_max, x:x + w]
    stats = sp.shape_stats(full_mask)
    return {"x": x, "crop": feather_crop, "mask": final_mask[:, :, 0] > 0,
            "full_mask": full_mask,   # untrimmed segmentation (after glue/card clean-up)
            "raw_mask": raw_mask,     # the segmentation as it came, for measured checks
            "record": {"bbox": [x, y, x + w, y_max], "area_px": stats["area"],
                       "length_px": round(stats["length"]), "width_px": round(stats["width"]),
                       "angle_deg": round(stats["angle"], 1),
                       "touches_border": bool(x <= 2 or y <= 2 or x + w >= img.shape[1] - 2
                                              or y + h >= img.shape[0] - 2),
                       "mask_cleanup": "; ".join(notes)}}


def equalized(crop: np.ndarray) -> np.ndarray:
    """The saved crop's look: luminance histogram-equalized."""
    img_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


class FeatherProcessor:
    def __init__(self) -> None:
        load_dotenv()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        # "sam3": SAM 3 finds and segments feathers from the text prompt in one model.
        # "dino_sam2": the original Grounding DINO boxes -> SAM 2.1 masks, kept for comparison.
        self.segmenter = os.getenv("FEATHER_SEGMENTER", "sam3")
        weights = os.getenv("FEATHER_SAM3_WEIGHTS", os.path.join(REPO_ROOT, "sam3.pt"))
        if self.segmenter == "sam3" and not os.path.exists(weights):
            # sam3.pt is gated on Hugging Face (facebook/sam3); until it is downloaded
            # next to the repo, keep working with the original detector.
            logging.warning("SAM 3 weights not found at %s; using dino_sam2", weights)
            self.segmenter = "dino_sam2"
        if self.segmenter == "sam3":
            from ultralytics.models.sam import SAM3SemanticPredictor

            self.sam3_weights = weights
            self._sam3_inter = None      # interactive predictor for sam3.refine, loaded on first use
            self.sam3 = SAM3SemanticPredictor(overrides={
                "conf": float(os.getenv("FEATHER_SAM3_CONF", "0.25")),
                "task": "segment",
                "mode": "predict",
                "model": weights,
                # SAM 3's native resolution; ultralytics' default 640 loses fine feather edges
                "imgsz": int(os.getenv("FEATHER_SAM3_IMGSZ", "1008")),
                "device": "mps" if torch.backends.mps.is_available() else "cpu",
                "save": False,
                "verbose": False,
            })
        elif self.segmenter == "dino_sam2":
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            self.dino_id = "IDEA-Research/grounding-dino-base"
            self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
            self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)
            self.sam_model = SAM("sam2.1_b.pt")
        else:
            raise ValueError(f"FEATHER_SEGMENTER must be sam3 or dino_sam2, not {self.segmenter!r}")

        self.has_vlm = False
        self.vlm_model = None
        self.vlm_processor = None
        self.vlm_generate = None
        self.vlm_model_id = os.getenv("FEATHER_VLM_MODEL", DEFAULT_VLM_MODEL)
        self.vlm_config = None
        # Specimen label table (src/label_extract.py): read from the original
        # slide before segmentation, independent of it. FEATHER_LABEL_VLM:
        # auto/local reuse this worker's MLX model when loaded, else the VLM
        # server (FEATHER_VLM_ENDPOINTS); server always uses the server.
        self.labels_enabled = os.getenv("FEATHER_ENABLE_LABELS", "1") == "1"
        self._label_client = None
        # Keep VLM opt-in so workers avoid expensive startup unless explicitly enabled.
        if os.getenv("FEATHER_ENABLE_VLM", "1") == "1":
            try:
                from mlx_vlm import generate as mlx_generate
                from mlx_vlm import load as mlx_load

                from mlx_vlm.utils import load_config

                self.vlm_model, self.vlm_processor = mlx_load(self.vlm_model_id)
                self.vlm_config = load_config(self.vlm_model_id)
                self.vlm_generate = mlx_generate
                self.has_vlm = True
            except Exception as exc:  # noqa: BLE001
                logging.warning("MLX VLM unavailable; continuing without QA model: %s", exc)

    def label_client(self):
        if self._label_client is None:
            from src import label_extract as lx

            mode = os.getenv("FEATHER_LABEL_VLM", "auto")
            if mode in ("auto", "local") and self.has_vlm:
                self._label_client = lx.MLXLocalClient(self.vlm_generate, self.vlm_model, self.vlm_processor,
                                                       self._chat_prompt, self.vlm_model_id)
            else:
                self._label_client = lx.VLMClient(lx.load_config()["vlm"])
        return self._label_client

    def extract_labels(self, image_path: str, output_dir: str):
        """Specimen label fields for one slide, or None when disabled. Never
        raises: a crash comes back as model_error fields so segmentation
        still runs and the slide is retried later, not exported as N/A."""
        if not self.labels_enabled:
            return None
        from src import label_extract as lx

        try:
            return lx.extract_for_pipeline(image_path, output_dir, self.label_client())
        except Exception as exc:  # noqa: BLE001
            logging.exception("Label extraction failed for %s", image_path)
            return lx.failed_result(image_path, exc)

    def _chat_prompt(self, instruction: str) -> str:
        from mlx_vlm.prompt_utils import apply_chat_template

        try:
            return apply_chat_template(self.vlm_processor, self.vlm_config, instruction,
                                       num_images=1, enable_thinking=False)
        except TypeError:  # templates without a thinking switch
            return apply_chat_template(self.vlm_processor, self.vlm_config, instruction, num_images=1)

    def _infer_metadata(self, image_path: str) -> tuple[str, str]:
        name = os.path.basename(image_path)
        stem = os.path.splitext(name)[0]

        bird_id = "UNKNOWN"
        bird_span: tuple[int, int] | None = None
        bird_match = re.search(r"(?<![A-Za-z0-9])([A-Z]{1,3})(\d{4,8})(?![A-Za-z0-9])", stem, flags=re.IGNORECASE)
        if bird_match:
            bird_id = f"{bird_match.group(1).upper()}{bird_match.group(2)}"
            bird_span = bird_match.span()

        date = "UNKNOWN"
        # Full date formats first.
        full_date = re.search(r"(?<!\d)(19|20)\d{2}[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?!\d)", stem)
        if full_date:
            date = full_date.group(0).replace("_", "-")
        else:
            compact_date = re.search(r"(?<!\d)((19|20)\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])(?!\d)", stem)
            if compact_date:
                yyyy = compact_date.group(1)
                mm = compact_date.group(3)
                dd = compact_date.group(4)
                date = f"{yyyy}-{mm}-{dd}"
            else:
                # Fall back to year if only year is present in filename.
                year_candidates = list(re.finditer(r"(?<!\d)(19|20)\d{2}(?!\d)", stem))
                for year_match in year_candidates:
                    ys, ye = year_match.span()
                    # Avoid taking year digits from inside the Bird_ID token (e.g., A1939).
                    if bird_span is not None and ys >= bird_span[0] and ye <= bird_span[1]:
                        continue
                    date = year_match.group(0)
                    break
        return bird_id, date

    def _cleanup(self) -> None:
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:  # noqa: BLE001
            pass

    def _enhance_for_dark_detection(self, img_pil: Image.Image) -> Image.Image:
        # Improve low-contrast dark feather visibility before DINO text grounding.
        bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)

    def _collect_boxes_with_scores(
        self,
        img_pil: Image.Image,
        prompt_text: str,
        score_thresh: float,
        max_box_area_ratio: float,
        shrink: float,
    ) -> list[list[float]]:
        inputs = self.dino_processor(images=img_pil, text=prompt_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.dino_model(**inputs)
        results = self.dino_processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            target_sizes=[img_pil.size[::-1]],
        )[0]

        img_area = img_pil.size[0] * img_pil.size[1]
        boxes_with_scores: list[list[float]] = []
        for score, box in zip(results["scores"], results["boxes"]):
            if float(score) <= score_thresh:
                continue
            b = box.tolist()
            width = b[2] - b[0]
            height = b[3] - b[1]
            if (width * height) > (img_area * max_box_area_ratio):
                continue
            b[0] += width * shrink
            b[1] += height * shrink
            b[2] -= width * shrink
            b[3] -= height * shrink
            boxes_with_scores.append([b[0], b[1], b[2], b[3], float(score)])
        return boxes_with_scores

    def _segment_sam3(self, image_path, img_pil, prompts, nms_iou, max_box_area_ratio, retry=False, conf=None):
        """SAM 3 concept segmentation: every instance matching the text prompts.
        Returns (boxes, masks, failure_reason, scores)."""
        if conf is not None and hasattr(self.sam3, "args"):
            self.sam3.args.conf = float(conf)
        texts = [p.rstrip(".") for p in prompts]
        sources = [image_path]
        if retry:
            # Contrast-enhanced pass for dark feathers, as the old detector did.
            enhanced = self._enhance_for_dark_detection(img_pil)
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            enhanced.save(tmp.name, quality=95)
            sources.append(tmp.name)
        boxes, scores, masks = [], [], []
        img_area = img_pil.size[0] * img_pil.size[1]
        try:
            for src in sources:
                self.sam3.set_image(src)
                for res in self.sam3(text=texts):
                    if res.masks is None or res.boxes is None:
                        continue
                    for b, sc, m in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy(),
                                        res.masks.data.cpu().numpy()):
                        if (b[2] - b[0]) * (b[3] - b[1]) > img_area * max_box_area_ratio:
                            continue
                        if conf is not None and float(sc) < float(conf):
                            continue
                        boxes.append([float(v) for v in b]); scores.append(float(sc)); masks.append(m)
        finally:
            if retry:
                os.unlink(sources[1])
        if not boxes:
            return [], [], "no_masks", []
        keep = torchvision.ops.nms(torch.tensor(boxes), torch.tensor(scores), nms_iou)
        return ([boxes[int(i)] for i in keep], [masks[int(i)] for i in keep], "",
                [scores[int(i)] for i in keep])

    def _interactive(self):
        """SAM 3's interactive predictor (box + points), loaded on first use; None without SAM 3."""
        if getattr(self, "sam3", None) is None:
            return None
        if getattr(self, "_sam3_inter", None) is None:
            from ultralytics.models.sam import SAM3Predictor

            self._sam3_inter = SAM3Predictor(overrides={
                "model": self.sam3_weights, "imgsz": int(os.getenv("FEATHER_SAM3_IMGSZ", "1008")),
                "task": "segment", "mode": "predict", "save": False, "verbose": False,
                "device": "mps" if torch.backends.mps.is_available() else "cpu"})
        return self._sam3_inter

    @staticmethod
    def _judge_feather(client, img, feather, crop_eq, jcfg, rel_area=None):
        """Per-feather checks on the saved crop's mask (bottom trimmed) or the untrimmed one
        (feather_judge.judge_mask); the panel is kept for review."""
        from src import feather_judge

        jmask = feather["mask"] if jcfg.get("judge_mask", "full") == "saved" else feather["full_mask"]
        res, panel = feather_judge.judge(client, img, jmask, crop_eq, jcfg, rel_area, feather.get("raw_mask"))
        return res, panel

    def _repair(self, client, img, feather, rec, others, jcfg, rcfg, trim, min_part, rel_area=None):
        """Repair attempts for one checked feather (src/feather_repair.py). Returns
        (feather, crop, record, panel) of the best attempt that fails fewer checks and
        nothing new, or None (the first mask stays; the record notes what was tried)."""
        from src import feather_judge, feather_repair

        old = feather_repair.fail_items(rec)
        tried, best = [], None
        for name, mask in feather_repair.candidates(rec, self._interactive(), img, feather["full_mask"], others, rcfg):
            f = make_feather(img, mask, trim, min_part, getattr(self, "_cleanup_cfg", None))
            if f is None:
                continue
            f["record"].update({k: feather["record"].get(k) for k in ("sam_score", "sam_masks", "postprocess")})
            crop_eq = equalized(f["crop"])
            res, panel = self._judge_feather(client, img, f, crop_eq, jcfg, rel_area)
            r2 = {"segment_status": "segmented", **f["record"], **{"judge_" + k: v for k, v in res.items()}}
            feather_judge.demote_unsupported(r2, jcfg)
            new = feather_repair.fail_items(r2) if res.get("status") == "ok" else None
            tried.append(f"{name}: " + ("check failed" if new is None else ("; ".join(sorted(new)) or "passed")))
            if new is not None and feather_repair.better(old, new) and (best is None or len(new) < best[0]):
                best = (len(new), f, crop_eq, r2, panel, name)
        if best is None:
            rec["repair"] = "tried, kept the first mask: " + (" | ".join(tried) or "no new mask")
            return None
        _, f, crop_eq, r2, panel, name = best
        r2["repair"] = f"{name} (before: {'; '.join(sorted(old))})"
        r2["repair_tried"] = " | ".join(tried)
        return f, crop_eq, r2, panel

    def _sam3_extras(self, image_path, img_pil, out, seg, prompts, nms_iou, max_box_area_ratio):
        """Optional steps after the text prompt (src/sam3_extras.py), each switched
        on in configs/segmentation.json: sam3.exemplar, then sam3.refine."""
        from src import sam3_extras

        ex, ref = seg.get("exemplar") or {}, seg.get("refine") or {}
        if getattr(self, "sam3", None) is None:      # no SAM 3 model loaded (e.g. a stand-in in tests)
            return out
        if ex.get("enabled"):
            out = sam3_extras.exemplar_pass(self.sam3, image_path, img_pil.size, prompts[0].rstrip("."), out, ex,
                                            nms_iou, max_box_area_ratio, seg.get("conf"))
        if ref.get("enabled") and not out[2]:
            out, info = sam3_extras.refine_pass(self._interactive(), image_path, out, ref)
            logging.info("sam3.refine %s", info)
        return out

    def _segment_dino_sam2(self, image_path, img_pil, prompts, score_thresh, nms_iou, max_box_area_ratio, shrink):
        """Original path: Grounding DINO boxes, then SAM 2.1 masks for them."""
        boxes_with_scores: list[list[float]] = []
        for p in prompts:
            boxes_with_scores.extend(
                self._collect_boxes_with_scores(
                    img_pil=img_pil,
                    prompt_text=p,
                    score_thresh=score_thresh,
                    max_box_area_ratio=max_box_area_ratio,
                    shrink=shrink,
                )
            )

        # Additional pass over contrast-enhanced image improves dark-feather recall.
        enhanced_img_pil = self._enhance_for_dark_detection(img_pil)
        for p in prompts:
            boxes_with_scores.extend(
                self._collect_boxes_with_scores(
                    img_pil=enhanced_img_pil,
                    prompt_text=p,
                    score_thresh=score_thresh,
                    max_box_area_ratio=max_box_area_ratio,
                    shrink=shrink,
                )
            )

        if not boxes_with_scores:
            return [], [], "no_boxes"

        boxes_tensor = torch.tensor([b[:4] for b in boxes_with_scores], dtype=torch.float32)
        scores_tensor = torch.tensor([b[4] for b in boxes_with_scores], dtype=torch.float32)
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_iou)
        boxes = [boxes_with_scores[int(i)][:4] for i in keep_indices]
        box_scores = [float(boxes_with_scores[int(i)][4]) for i in keep_indices]

        if not boxes:
            return [], [], "no_boxes_after_nms"

        sam_results = self.sam_model(image_path, bboxes=boxes, device=self.device, verbose=False)
        sam_res = sam_results[0]
        if sam_res.masks is None:
            return [], [], "no_masks", []
        return boxes, list(sam_res.masks.data.cpu().numpy()), "", box_scores

    def process_image(self, image_path: str, output_dir: str, profile: str = "default",
                      labels=None, options: dict | None = None) -> ProcessResult:
        """`labels`: this slide's label result (extract_labels); when given, the
        saved crops get feather IDs from its bird ID and year. `options`:
        {'segmentation': {...}} overrides configs/segmentation.json for this slide."""
        from src import feather_judge, feather_repair
        from src import segment_postprocess as sp

        cfg = load_seg_config((options or {}).get("segmentation"))
        bird_id, date = self._infer_metadata(image_path)
        metadata_source = "filename"

        try:
            img_pil = Image.open(image_path).convert("RGB")
            seg = cfg[self.segmenter]
            prompts = list(seg["prompts"])
            nms_iou, max_box_area_ratio = seg["nms_iou"], seg["max_box_area_ratio"]
            score_thresh = seg.get("score_thresh", 0.25)
            if profile == "strict_retry":
                # In retry mode, broaden detector recall for hard/dark feathers.
                score_thresh = min(score_thresh, 0.2)
                nms_iou = min(nms_iou, 0.2)
                prompts.extend(seg.get("retry_prompts", []))

            if self.segmenter == "sam3":
                out = self._segment_sam3(image_path, img_pil, prompts, nms_iou, max_box_area_ratio,
                                         retry=(profile == "strict_retry"), conf=seg.get("conf"))
            else:
                out = self._segment_dino_sam2(image_path, img_pil, prompts, score_thresh, nms_iou,
                                              max_box_area_ratio, seg.get("shrink", 0.02))
            if self.segmenter == "sam3" and not out[2]:
                out = self._sam3_extras(image_path, img_pil, out, seg, prompts, nms_iou, max_box_area_ratio)
            boxes, mask_list, reason = out[:3]
            mask_scores = list(out[3]) if len(out) > 3 else [1.0] * len(mask_list)
            if reason:
                return ProcessResult(
                    image_path=image_path,
                    success=False,
                    reason=reason,
                    metadata_source=metadata_source,
                    bird_id=bird_id,
                    date=date,
                )

            img = cv2.imread(image_path)
            feathers: list[dict] = []
            full_masks = [cv2.resize(m.astype(np.float32), (img.shape[1], img.shape[0])) > 0.5 for m in mask_list]
            # one mask per feather: merge slivers/split pieces, drop contained duplicates
            pieces = sp.clean_masks(full_masks, mask_scores, cfg.get("postprocess"))
            trim = float(cfg["crop"].get("trim_bottom_frac", 0.12))
            min_part = sp.DEFAULTS["min_area_frac"] * img.shape[0] * img.shape[1] * 0.25

            cleanup = cfg["crop"].get("cleanup") or {}
            self._cleanup_cfg = cleanup       # repair attempts are cleaned the same way
            for piece in pieces:
                f = make_feather(img, piece.mask, trim, min_part, cleanup)
                if f is None:
                    continue
                f["record"].update({"sam_score": round(float(piece.score), 3), "sam_masks": len(piece.sources),
                                    "postprocess": "; ".join(piece.notes)})
                feathers.append(f)

            if not feathers:
                return ProcessResult(
                    image_path=image_path,
                    success=False,
                    reason="no_feathers_after_sam",
                    metadata_source=metadata_source,
                    bird_id=bird_id,
                    date=date,
                )

            os.makedirs(output_dir, exist_ok=True)
            feathers.sort(key=lambda item: item["x"])
            from src import output_names as on

            label_date = None
            if labels is not None and labels.fields.get("date") is not None \
                    and labels.fields["date"].status != "model_error":
                label_date = labels.fields["date"].value
            # 'CE01168 2008' / 'CE01168_2008'; full label date for same-year slides,
            # plus the image number for same-day slides (see output_names.py)
            input_dir = os.path.dirname(image_path) or "."
            siblings = os.listdir(input_dir) if os.path.isdir(input_dir) else []
            sibling_dates = None
            if labels is not None and label_date and on.shares_id_year(image_path, siblings):
                from src import label_extract as lx

                sibling_dates = {s: lx.slide_date(os.path.join(input_dir, s), output_dir, self.label_client())
                                 for s in on.same_year_siblings(image_path, siblings)}
            names = on.output_names(image_path, siblings, date=label_date, sibling_dates=sibling_dates)
            source_name = os.path.basename(image_path)
            for old in os.listdir(output_dir):
                # crops from an earlier attempt at this slide (new or old names)
                if (("_Feather_" in old or "_Fragment_" in old) and old.endswith(".jpg")
                        and on.belongs_to(old, source_name, label_date)):
                    try:
                        os.remove(os.path.join(output_dir, old))
                    except OSError:
                        pass
            mask_dir = os.path.join(output_dir, "feather_masks")
            for old in (os.listdir(mask_dir) if os.path.isdir(mask_dir) else []):
                if old.endswith(".png") and on.belongs_to(old[:-4] + ".jpg", source_name, label_date):
                    try:
                        os.remove(os.path.join(mask_dir, old))
                    except OSError:
                        pass
            feathers_to_save = feathers[: int(cfg["crop"].get("max_feathers", 8))]
            jcfg, icfg = cfg.get("feather_judge", {}), cfg.get("inventory", {})
            vlm_client = self.label_client() if (jcfg.get("enabled") or icfg.get("enabled")) else None
            qc_dir = os.path.join(output_dir, "feather_qc")
            # Missed feathers: the model counts every feather on the numbered overlay. Counting
            # overlapping feathers is unreliable, so its answer only adds review rows (no letter,
            # crop or feather ID) and never changes the letters or names of the segmented feathers.
            inv_items: list[dict] = []
            if icfg.get("enabled") and vlm_client is not None and feathers_to_save:
                from src import feather_inventory

                inv_items, inventory, inv_img = feather_inventory.check(
                    vlm_client, img, [f["full_mask"] for f in feathers_to_save], icfg)
                os.makedirs(qc_dir, exist_ok=True)
                inv_name = f"{names['base']}_inventory.jpg"
                inv_img.save(os.path.join(qc_dir, inv_name), quality=85)
                inventory["inventory_image"] = f"feather_qc/{inv_name}"
            else:
                inventory = {"inventory_status": "disabled", "feathers_segmented": len(feathers_to_save),
                             "feathers_possibly_missed": 0}

            overlay = img.copy()
            green = np.zeros_like(overlay)
            green[:, :, 1] = 255  # Green mask
            for piece in pieces:
                overlay = np.where(piece.mask[:, :, None], cv2.addWeighted(overlay, 0.5, green, 0.5, 0), overlay)
            for f in feathers:
                x1, y1, x2, y2 = f["record"]["bbox"]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding boxes
            for it in inv_items:
                if it["kind"] == "missed":   # where the model saw a feather without a segment (review flag)
                    xm = int(it["x"])
                    cv2.line(overlay, (xm, int(img.shape[0] * 0.25)), (xm, int(img.shape[0] * 0.9)), (0, 0, 255), 6)
                    cv2.putText(overlay, "possible missed", (max(0, xm - 200), int(img.shape[0] * 0.24)),
                                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 5)

            bbox_filename = on.overlay_name(names)
            bbox_filepath = os.path.join(output_dir, bbox_filename)
            cv2.imwrite(bbox_filepath, overlay)
            # The slide-level VLM judge (metadata + one 1-10 overlay score) is retired: the
            # label extractor owns bird ID and date, and the inventory plus the per-feather
            # checks below cover what it asked.

            # Each piece is checked first, then named: pieces the clean-up marked small or narrow
            # that the model also sees as a fragment get no letter or feather ID (review rows,
            # crop '<prefix>_Fragment_<n>.jpg'); the feathers are lettered A, B, C ... left to right.
            judge_client = vlm_client if jcfg.get("enabled") else None
            rcfg = jcfg.get("repair") or {}
            repairs_left = int(rcfg.get("max_per_slide", 6))
            checked = []
            med_area = float(np.median([f["raw_mask"].sum() for f in feathers_to_save])) if feathers_to_save else 0
            for fi, feather in enumerate(feathers_to_save):
                crop_eq = equalized(feather["crop"])
                rec = {"segment_status": "segmented", **feather["record"]}
                panel = None
                if judge_client is not None:
                    rel_area = feather["raw_mask"].sum() / med_area if med_area else None
                    res, panel = self._judge_feather(judge_client, img, feather, crop_eq, jcfg, rel_area)
                    rec.update({"judge_" + k: v for k, v in res.items()})
                    feather_judge.demote_unsupported(rec, jcfg)
                    gcfg = jcfg.get("regrow") or {}
                    if gcfg.get("enabled") and rec.get("judge_status") == "ok" and self._interactive() is not None:
                        others = [f["raw_mask"] for j, f in enumerate(feathers_to_save) if j != fi]
                        g = feather_repair.regrow_check(self._interactive(), img, feather["raw_mask"], others, gcfg)
                        rec.update({"judge_" + k: v for k, v in g.items() if k != "regrow_cut"})
                        feather_judge.add_cut_sides(rec, g.get("regrow_cut") or [], jcfg)
                        jmask = feather["mask"] if jcfg.get("judge_mask", "full") == "saved" else feather["full_mask"]
                        feather_judge.demote_board_cuts(rec, img, jmask, jcfg)
                    if rcfg.get("enabled") and repairs_left > 0 and feather_repair.worth_repairing(rec):
                        # second sweep: new masks for a fixable failure, kept only if they check better
                        repairs_left -= 1
                        others = [f["full_mask"] for j, f in enumerate(feathers_to_save) if j != fi]
                        better = self._repair(judge_client, img, feather, rec, others, jcfg, rcfg, trim, min_part,
                                              rel_area)
                        if better is not None:
                            feather, crop_eq, rec, panel = better
                else:
                    rec["judge_status"] = "disabled"
                if feather_judge.is_fragment(rec):
                    rec["segment_status"] = "fragment"
                checked.append((rec, crop_eq, panel, feather))
            records, fragments, labels_by_index = [], [], {}
            save_masks = bool(cfg["crop"].get("save_masks", True))
            for i, (rec, crop_eq, panel, feather) in enumerate(checked):
                if rec["segment_status"] == "fragment":
                    k = len(fragments) + 1
                    seq, filename = f"fragment_{k}", on.fragment_name(names, k)
                else:
                    seq, filename = on.seq_letter(len(records)), on.crop_name(names, len(records))
                labels_by_index[i] = seq
                cv2.imwrite(os.path.join(output_dir, filename), crop_eq)
                rec = {"seq": seq, "crop": filename, **rec}
                if save_masks:
                    # the segmentation as it came (before glue/card clean-up, which is recomputed from it),
                    # so checks can be re-run later on exactly this segmentation
                    os.makedirs(os.path.join(output_dir, "feather_masks"), exist_ok=True)
                    mask_name = f"feather_masks/{filename[:-4]}.png"
                    cv2.imwrite(os.path.join(output_dir, mask_name), feather["raw_mask"].astype(np.uint8) * 255,
                                [cv2.IMWRITE_PNG_COMPRESSION, 9])
                    rec["mask_image"] = mask_name
                if panel is not None:
                    os.makedirs(qc_dir, exist_ok=True)
                    qc_name = filename[:-4] + "_qc.jpg"
                    panel.save(os.path.join(qc_dir, qc_name), quality=85)
                    rec["qc_image"] = f"feather_qc/{qc_name}"
                if rec["segment_status"] == "fragment":
                    rec["feather_id"] = ""
                    rec["needs_review"] = ("fragment: small or narrow piece the model sees as part of a feather "
                                           "(no letter or feather ID; check the QC image)")
                    fragments.append(rec)
                else:
                    rec["needs_review"] = feather_judge.needs_review(rec)
                    rec["review_uncertain"] = feather_judge.uncertain_review(rec)
                    records.append(rec)
            feather_ids = records
            if labels is not None:
                from src import label_extract as lx

                ids = lx.assign_feather_ids(labels, [r["crop"] for r in records], names["period"],
                                            seqs=[r["seq"] for r in records])
                feather_ids = [{**rec, **fid} for rec, fid in zip(records, ids)]
            feather_ids = feather_ids + fragments
            if inv_items:
                from src import feather_inventory

                feather_ids = feather_ids + [
                    {"feather_id": "", **row} for row in feather_inventory.review_rows(
                        inv_items, inventory, labels_by_index)]
            for rec in feather_ids:
                rec.update({k: v for k, v in inventory.items() if k != "inventory_raw"})
            if labels is not None:
                labels.feathers = feather_ids

            # Slide-level QA, derived from the checks (kept in the existing vlm_* fields of
            # image_stats so the dashboards keep working): mean derived feather score,
            # inventory complete, other material / merged feathers in any kept feather.
            checks = feather_judge.slide_checks(feather_ids)
            vlm_score = checks["score_mean"]
            failing = [f"{r['seq']}: {', '.join(feather_judge.failed_checks(r))}" for r in records
                       if feather_judge.failed_checks(r)]
            vlm_notes = "; ".join(failing) or ("all feathers passed" if checks["judge_errors"] == 0 and records
                                               else "")
            if fragments:
                vlm_notes = (vlm_notes + "; " if vlm_notes else "") + f"{len(fragments)} fragment(s) set aside"
            if checks["possible_missed"]:
                vlm_notes = (vlm_notes + "; " if vlm_notes else "") + f"{checks['possible_missed']} possible missed"
            return ProcessResult(
                image_path=image_path,
                success=True,
                feathers_saved=len(records),
                vlm_score=vlm_score,
                metadata_source=metadata_source,
                bird_id=bird_id,
                date=date,
                vlm_all_feathers_covered=checks["all_feathers_covered"],
                vlm_background_leakage_detected=checks["other_material"],
                vlm_green_boxes_grouped_feathers=checks["merged_feathers"],
                vlm_notes=vlm_notes,
                processing_profile=profile,
                feather_ids=feather_ids,
                qa_checks=checks,
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception("Failed to process image %s", image_path)
            return ProcessResult(
                image_path=image_path,
                success=False,
                reason=str(exc),
                metadata_source=metadata_source,
                bird_id=bird_id,
                date=date,
                processing_profile=profile,
            )
        finally:
            self._cleanup()
