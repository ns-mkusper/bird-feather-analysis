import gc
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torchvision
from dotenv import load_dotenv
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
from ultralytics import SAM


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
    processing_profile: str = "default"


class FeatherProcessor:
    def __init__(self) -> None:
        load_dotenv()
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.dino_id = "IDEA-Research/grounding-dino-base"
        self.dino_processor = AutoProcessor.from_pretrained(self.dino_id)
        self.dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(self.dino_id).to(self.device)
        self.sam_model = SAM("sam2.1_b.pt")

        self.has_vlm = False
        self.vlm_model = None
        self.vlm_processor = None
        self.vlm_generate = None
        self.vlm_scoring_enabled = os.getenv("FEATHER_ENABLE_VLM_SCORING", "0") == "1"
        self.vlm_metadata_enabled = os.getenv("FEATHER_ENABLE_VLM_METADATA", "1") == "1"
        self.vlm_model_id = os.getenv("FEATHER_VLM_MODEL", "mlx-community/Qwen3-VL-8B-Instruct-4bit")
        # Keep VLM opt-in so workers avoid expensive startup unless explicitly enabled.
        if os.getenv("FEATHER_ENABLE_VLM", "0") == "1":
            try:
                from mlx_vlm import generate as mlx_generate
                from mlx_vlm import load as mlx_load

                self.vlm_model, self.vlm_processor = mlx_load(self.vlm_model_id)
                self.vlm_generate = mlx_generate
                self.has_vlm = True
            except Exception as exc:  # noqa: BLE001
                logging.warning("MLX VLM unavailable; continuing without QA model: %s", exc)

    def _normalize_vlm_metadata(self, raw_bird: str, raw_date: str) -> tuple[str, str]:
        bird = "UNKNOWN"
        bird_m = re.search(r"([A-Z]{1,3})\s*[-_ ]?\s*(\d{4,8})", raw_bird, flags=re.IGNORECASE)
        if bird_m:
            bird = f"{bird_m.group(1).upper()}{bird_m.group(2)}"

        date = "UNKNOWN"
        date_full = re.search(r"(19|20)\d{2}[-_/](0[1-9]|1[0-2])[-_/](0[1-9]|[12]\d|3[01])", raw_date)
        if date_full:
            date = date_full.group(0).replace("_", "-").replace("/", "-")
        else:
            date_compact = re.search(r"((19|20)\d{2})(0[1-9]|1[0-2])([0-2]\d|3[01])", raw_date)
            if date_compact:
                date = f"{date_compact.group(1)}-{date_compact.group(3)}-{date_compact.group(4)}"
            else:
                year_only = re.search(r"(19|20)\d{2}", raw_date)
                if year_only:
                    date = year_only.group(0)
        return bird, date

    def _vlm_judge(self, image_path: str) -> dict:
        out = {
            "Bird_ID": "UNKNOWN",
            "Date": "UNKNOWN",
            "quality_score_1_to_10": None,
            "all_feathers_covered": None,
            "background_leakage_detected": None,
            "green_boxes_grouped_feathers": None,
        }
        if not (self.has_vlm and self.vlm_generate is not None and (self.vlm_scoring_enabled or self.vlm_metadata_enabled)):
            return out
        prompt = (
            "<|im_start|>system\nYou are a precise computer vision quality analyst. Return ONLY valid JSON.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\n"
            "Given this feather slide image/overlay, extract specimen metadata and segmentation quality.\n"
            "Return exactly JSON with keys:\n"
            '{"Bird_ID":"A1234 or UNKNOWN","Date":"YYYY-MM-DD or YYYY or UNKNOWN","quality_score_1_to_10":0-10,'
            '"all_feathers_covered":true/false,"background_leakage_detected":true/false,"green_boxes_grouped_feathers":true/false}'
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        try:
            output = self.vlm_generate(
                self.vlm_model,
                self.vlm_processor,
                prompt=prompt,
                image=[image_path],
                max_tokens=180,
            )
            text = getattr(output, "text", str(output))
            match = re.search(r"\{.*?\}", text, re.DOTALL)
            if match:
                obj = json.loads(match.group(0))
                bird, date = self._normalize_vlm_metadata(str(obj.get("Bird_ID", "")), str(obj.get("Date", "")))
                out["Bird_ID"] = bird
                out["Date"] = date
                score = obj.get("quality_score_1_to_10")
                if isinstance(score, (int, float)):
                    out["quality_score_1_to_10"] = float(score)
                for key in ("all_feathers_covered", "background_leakage_detected", "green_boxes_grouped_feathers"):
                    val = obj.get(key)
                    if isinstance(val, bool):
                        out[key] = val
                return out
            num = re.search(r"([0-9]+(?:\\.[0-9]+)?)", text)
            if num:
                out["quality_score_1_to_10"] = float(num.group(1))
            return out
        except Exception as exc:  # noqa: BLE001
            logging.warning("VLM judge failed for %s: %s", image_path, exc)
            return out

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

    def process_image(self, image_path: str, output_dir: str, profile: str = "default") -> ProcessResult:
        bird_id, date = self._infer_metadata(image_path)
        metadata_source = "filename"
        vlm_judge = {
            "quality_score_1_to_10": None,
            "all_feathers_covered": None,
            "background_leakage_detected": None,
            "green_boxes_grouped_feathers": None,
            "Bird_ID": "UNKNOWN",
            "Date": "UNKNOWN",
        }

        try:
            img_pil = Image.open(image_path).convert("RGB")
            score_thresh = 0.25
            nms_iou = 0.3
            max_box_area_ratio = 0.45
            shrink = 0.02
            prompts = ["bird feather."]
            if profile == "strict_retry":
                # In retry mode, broaden detector recall for hard/dark feathers.
                score_thresh = 0.2
                nms_iou = 0.2
                max_box_area_ratio = 0.45
                shrink = 0.02
                prompts.extend(["dark bird feather.", "black feather."])

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
                return ProcessResult(
                    image_path=image_path,
                    success=False,
                    reason="no_boxes",
                    metadata_source=metadata_source,
                    bird_id=bird_id,
                    date=date,
                )

            boxes_tensor = torch.tensor([b[:4] for b in boxes_with_scores], dtype=torch.float32)
            scores_tensor = torch.tensor([b[4] for b in boxes_with_scores], dtype=torch.float32)
            keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, nms_iou)
            boxes = [boxes_with_scores[int(i)][:4] for i in keep_indices]

            if not boxes:
                return ProcessResult(
                    image_path=image_path,
                    success=False,
                    reason="no_boxes_after_nms",
                    metadata_source=metadata_source,
                    bird_id=bird_id,
                    date=date,
                )

            sam_results = self.sam_model(image_path, bboxes=boxes, device=self.device, verbose=False)
            sam_res = sam_results[0]
            if sam_res.masks is None:
                return ProcessResult(
                    image_path=image_path,
                    success=False,
                    reason="no_masks",
                    metadata_source=metadata_source,
                    bird_id=bird_id,
                    date=date,
                )

            img = cv2.imread(image_path)
            feathers: list[dict] = []

            mask_list = sam_res.masks.data.cpu().numpy()
            for mask in mask_list:
                mask_resized = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue

                contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(contour)
                y_max = y + h - int(h * 0.12)

                final_mask = np.zeros_like(img)
                cv2.drawContours(final_mask, [contour], -1, (255, 255, 255), -1)
                final_mask[y_max:, :] = 0

                clean_feather = cv2.bitwise_and(img, final_mask)
                feather_crop = clean_feather[y:y_max, x : x + w]
                feathers.append({"x": x, "crop": feather_crop})

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
            source_stem = os.path.splitext(os.path.basename(image_path))[0]
            for old in os.listdir(output_dir):
                if old.startswith(f"{source_stem}_") and "_Feather_" in old and old.endswith(".jpg"):
                    try:
                        os.remove(os.path.join(output_dir, old))
                    except OSError:
                        pass
            # Create a simple visual QA overlay for VLM judgment and notebook preview.
            overlay = img.copy()
            for mask in mask_list:
                mask_resized = cv2.resize(mask.astype(np.float32), (img.shape[1], img.shape[0]))
                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                green = np.zeros_like(overlay)
                green[:, :, 1] = 255  # Green mask
                overlay = np.where(binary_mask[:, :, None] == 1, cv2.addWeighted(overlay, 0.5, green, 0.5, 0), overlay)
            for b in boxes:
                x1, y1, x2, y2 = [int(v) for v in b]
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bounding boxes
            
            bbox_filename = f"{source_stem}_BoundingBoxes.jpg"
            bbox_filepath = os.path.join(output_dir, bbox_filename)
            cv2.imwrite(bbox_filepath, overlay)

            judge_path = bbox_filepath
            if self.has_vlm and (self.vlm_scoring_enabled or self.vlm_metadata_enabled):
                vlm_judge = self._vlm_judge(judge_path)

            if bird_id == "UNKNOWN" and vlm_judge.get("Bird_ID") != "UNKNOWN":
                bird_id = str(vlm_judge["Bird_ID"])
                metadata_source = "vlm_fallback"
            if date == "UNKNOWN" and vlm_judge.get("Date") != "UNKNOWN":
                date = str(vlm_judge["Date"])
                metadata_source = "vlm_fallback"
            if (bird_id == "UNKNOWN" or date == "UNKNOWN") and metadata_source != "vlm_fallback":
                metadata_source = "unknown"

            feathers_to_save = feathers[:5]
            for idx, feather in enumerate(feathers_to_save, start=1):
                crop = feather["crop"]
                img_yuv = cv2.cvtColor(crop, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                crop_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

                filename = f"{source_stem}_{bird_id}_{date}_Feather_{idx}.jpg"
                cv2.imwrite(os.path.join(output_dir, filename), crop_eq)

            vlm_score = (
                float(vlm_judge["quality_score_1_to_10"])
                if self.vlm_scoring_enabled and isinstance(vlm_judge.get("quality_score_1_to_10"), (int, float))
                else None
            )
            return ProcessResult(
                image_path=image_path,
                success=True,
                feathers_saved=len(feathers_to_save),
                vlm_score=vlm_score,
                metadata_source=metadata_source,
                bird_id=bird_id,
                date=date,
                vlm_all_feathers_covered=vlm_judge.get("all_feathers_covered"),
                vlm_background_leakage_detected=vlm_judge.get("background_leakage_detected"),
                vlm_green_boxes_grouped_feathers=vlm_judge.get("green_boxes_grouped_feathers"),
                processing_profile=profile,
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
