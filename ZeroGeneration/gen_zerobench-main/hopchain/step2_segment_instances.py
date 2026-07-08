#!/usr/bin/env python3
"""
HopChain Step 2 — Segment Object Instances with SAM3

For each category string returned by Step 1, run SAM3 text-prompted segmentation
to obtain per-instance masks, bounding boxes, and cropped patches.

Instance dict schema:
    {
        "instance_id": "person_0",   # "{category}_{index}"
        "category":    "person",
        "bbox":        [x0, y0, x1, y1],   # absolute pixels, XYXY
        "score":       float,
        "crop":        PIL.Image,           # padded crop from original
        "mask":        np.ndarray,          # bool H×W
    }
"""

from __future__ import annotations

import itertools
import random
import sys
from typing import Optional

import numpy as np
from PIL import Image


# ── SAM3 model path ───────────────────────────────────────────────────────────

_SAM3_MODEL_PATH = (
    "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/model/sam3"
)


def load_sam3_model(device: str = "cuda"):
    """
    Load SAM3 image model and processor via HuggingFace Transformers API.

    Returns:
        (model, processor) tuple ready to use with segment_instances().
    """
    import torch
    from transformers import Sam3Model, Sam3Processor  # type: ignore

    model = Sam3Model.from_pretrained(_SAM3_MODEL_PATH).to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained(_SAM3_MODEL_PATH)
    return model, processor


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[hopchain/step2] {msg}", flush=True)


# ── Core segmentation function ────────────────────────────────────────────────

def segment_instances(
    image: Image.Image,
    categories: list[str],
    model,
    processor,
    min_score: float = 0.4,
    max_per_category: int = 5,
    max_total: int = 20,
    crop_pad_px: int = 4,
    min_area_frac: float = 0.005,
) -> list[dict]:
    """
    Run SAM3 text-prompted segmentation for each category using the HF API.

    Args:
        image:            PIL image to segment.
        categories:       List of category name strings.
        model:            Sam3Model from load_sam3_model().
        processor:        Sam3Processor from load_sam3_model().
        min_score:        Minimum confidence score to keep an instance.
        max_per_category: Maximum instances to keep per category.
        max_total:        Hard cap on total instances returned.
        crop_pad_px:      Pixel padding added to each side of the bbox crop.
        min_area_frac:    Minimum bbox area as fraction of image area (skip smaller).

    Returns:
        List of instance dicts sorted by score descending.
    """
    import torch

    img_w, img_h = image.size
    img_area = img_w * img_h
    min_bbox_area = img_area * min_area_frac

    device = next(model.parameters()).device

    instances: list[dict] = []
    category_counters: dict[str, int] = {}

    for category in categories:
        if len(instances) >= max_total:
            break

        try:
            inputs = processor(
                images=image,
                text=category,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=min_score,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

            masks  = results.get("masks")   # [N, H, W] bool tensors
            boxes  = results.get("boxes")   # [N, 4] XYXY absolute
            scores = results.get("scores")  # [N]

        except Exception as e:
            _log(f"segment '{category}' failed: {e!r}")
            continue

        if masks is None or len(masks) == 0:
            _log(f"category='{category}': no detections")
            continue

        # Convert to numpy
        if hasattr(masks,  "cpu"): masks  = masks.cpu().numpy()
        if hasattr(boxes,  "cpu"): boxes  = boxes.cpu().numpy()
        if hasattr(scores, "cpu"): scores = scores.cpu().numpy()

        masks  = np.asarray(masks)
        boxes  = np.asarray(boxes)
        scores = np.asarray(scores).flatten()

        n_found = len(scores)
        _log(f"category='{category}': {n_found} raw detections")

        order = np.argsort(-scores)
        cat_count = category_counters.get(category, 0)

        for idx in order:
            if cat_count >= max_per_category:
                break
            if len(instances) >= max_total:
                break

            score = float(scores[idx])

            x0, y0, x1, y1 = boxes[idx].tolist()
            x0 = max(0.0, float(x0))
            y0 = max(0.0, float(y0))
            x1 = min(float(img_w), float(x1))
            y1 = min(float(img_h), float(y1))

            bbox_area = (x1 - x0) * (y1 - y0)
            if bbox_area < min_bbox_area:
                _log(f"  skip {category}#{idx}: area {bbox_area:.0f} < min {min_bbox_area:.0f}")
                continue

            raw_mask = masks[idx]
            if raw_mask.ndim == 3:
                raw_mask = raw_mask[0]
            bool_mask = raw_mask.astype(bool)

            cx0 = max(0, int(x0) - crop_pad_px)
            cy0 = max(0, int(y0) - crop_pad_px)
            cx1 = min(img_w, int(x1) + crop_pad_px)
            cy1 = min(img_h, int(y1) + crop_pad_px)
            crop = image.crop((cx0, cy0, cx1, cy1))

            instance_id = f"{category}_{cat_count}"
            instances.append({
                "instance_id": instance_id,
                "category":    category,
                "bbox":        [int(x0), int(y0), int(x1), int(y1)],
                "score":       score,
                "crop":        crop,
                "mask":        bool_mask,
            })
            cat_count += 1
            _log(
                f"  + {instance_id} score={score:.3f} "
                f"bbox=[{int(x0)},{int(y0)},{int(x1)},{int(y1)}]"
            )

        category_counters[category] = cat_count

    instances.sort(key=lambda d: d["score"], reverse=True)
    return instances[:max_total]


# ── Combination selector ──────────────────────────────────────────────────────

def select_instance_combinations(
    instances: list[dict],
    combo_size: int = 4,
    max_combos: int = 10,
) -> list[list[dict]]:
    """
    Sample combinations of *combo_size* instances from *instances*.

    Preference order:
      1. Combinations with at least 3 distinct categories (diverse).
      2. All remaining combinations, shuffled.

    Args:
        instances:  Full list of instance dicts from segment_instances().
        combo_size: Number of instances per combination.
        max_combos: Maximum number of combinations returned.

    Returns:
        List of combinations; each combination is a list of instance dicts.
    """
    if len(instances) < combo_size:
        _log(
            f"select_instance_combinations: only {len(instances)} instances, "
            f"need {combo_size} — returning single combo if possible"
        )
        return [instances] if instances else []

    all_combos = list(itertools.combinations(instances, combo_size))
    random.shuffle(all_combos)

    # Partition into diverse (≥3 categories) and non-diverse
    diverse:     list[list[dict]] = []
    non_diverse: list[list[dict]] = []

    for combo in all_combos:
        n_cats = len({inst["category"] for inst in combo})
        if n_cats >= 3:
            diverse.append(list(combo))
        else:
            non_diverse.append(list(combo))

    ordered = diverse + non_diverse
    selected = ordered[:max_combos]

    _log(
        f"select_instance_combinations: {len(all_combos)} total, "
        f"{len(diverse)} diverse → returning {len(selected)}"
    )
    return selected


# ── CLI (quick smoke-test) ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HopChain Step 2: Segment object instances with SAM3"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--categories", nargs="+", default=["person", "car", "dog"],
        help="Categories to segment",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_score", type=float, default=0.4)
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    _log("Loading SAM3 model…")
    sam_model, sam_processor = load_sam3_model(device=args.device)
    _log("Segmenting…")
    insts = segment_instances(img, args.categories, sam_model, sam_processor,
                              min_score=args.min_score)
    _log(f"Found {len(insts)} instances:")
    for inst in insts:
        _log(f"  {inst['instance_id']}: score={inst['score']:.3f} bbox={inst['bbox']}")

    combos = select_instance_combinations(insts, combo_size=4, max_combos=5)
    _log(f"Generated {len(combos)} combinations")
