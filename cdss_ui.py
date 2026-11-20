# -*- coding: utf-8 -*-
"""Автоматически сгенерировано из Ячейки 3 (кроссплатформенный UI и пакетный аудит)."""

import os
import tempfile
from pathlib import Path
from compat import configure_console_utf8, amp_autocast, imread_unicode
from config import (
    PUBLIC_DEMO_20_GDRIVE_URL,
    PUBLIC_DEMO_20_GDRIVE_FILES,
    AUTO_RUN_DEMO_20_ON_STARTUP,
)
from cdss_core import *  # noqa: F401,F403
from cdss_core import (
    _compute_canonical_parenchyma_mask,
    _evaluate_skull_and_orbit_artifact_penalty,
    _is_glioma_csf_or_ventricle_artifact,
    _safe_torch_load,
)

configure_console_utf8()

# -*- coding: utf-8 -*-
"""
===========================================================================================
ЯЧЕЙКА 3: ИНТЕРАКТИВНЫЙ GRADIO UI + МАССОВЫЙ АУДИТ (СИНХРОНИЗИРОВАНА С ЯЧЕЙКОЙ 2)
===========================================================================================
  1. Полная синхронизация с Ячейкой 2 + автопроверка _verify_cell2_synced().
  2. Порог сегментации по умолчанию: tau = 0.80 (DEFAULT_SEG_THRESHOLD).
  3. Расчёт и отображение строго 2D-площади (мм²), без объёма и без дат.
  4. Отрисовка Слоя 4 (неуверенная зона / граница с наименьшей вероятностью) пунктиром.
  5. Автономный запуск на тестовом ПК (T4 / L4 / A100 / CPU) без drive.mount().
"""

import os
import re
import math
import time
import hashlib
import zipfile
import shutil
import tempfile
import traceback
from pathlib import Path
from dataclasses import asdict
from typing import Dict, List, Tuple, Optional, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

import torch
import torch.nn.functional as F
import gradio as gr

PHOTOS_PER_CARD = 15  # Строго 15 фото на каждой карточке (сетка 3 x 5)
DEMO_20_MAX_FILES = 20  # Плашка на все 20 выбранных МРТ (сетка 4 x 5)
BATCH_CARDS_DIR = BASE_DIR / "interactive_batch_audit_v8_512"
DEMO_20_DIR = BASE_DIR / "demo_20_showcase_512"
BATCH_CARDS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_20_DIR.mkdir(parents=True, exist_ok=True)

if "PUBLIC_DEMO_20_GDRIVE_URL" not in globals():
    PUBLIC_DEMO_20_GDRIVE_URL = ""
if "PUBLIC_DEMO_20_GDRIVE_FILES" not in globals():
    PUBLIC_DEMO_20_GDRIVE_FILES = []
if "AUTO_RUN_DEMO_20_ON_STARTUP" not in globals():
    AUTO_RUN_DEMO_20_ON_STARTUP = True

AUTO_RUN_4CLASS_ON_STARTUP = False
DEFAULT_MAX_SLICES_PER_CLASS_STARTUP = 250

CLASS_SHORT_RU = {
    "glioma": "Глиома",
    "meningioma": "Менингиома",
    "no_tumor": "Норма",
    "pituitary": "Гипофиз",
    "unknown": "Не указан",
}


# ==============================================================================
# 1. ПОИСК ЧЕКПОИНТОВ И ПРОВЕРКА СИНХРОНИЗАЦИИ С ЯЧЕЙКОЙ 2
# ==============================================================================
def _resolve_best_checkpoint_dir(default_dir: Path) -> Path:
    candidate_dirs = [
        default_dir,
        BASE_DIR / "checkpoints_512",
        BASE_DIR / "drive" / "MyDrive" / "checkpoints_512",
        Path("/content/checkpoints_512"),
        Path("/content/drive/MyDrive/checkpoints_512"),
    ]
    for c_dir in candidate_dirs:
        mt_file = c_dir / "multitask_512_best.pth"
        if mt_file.exists() and mt_file.stat().st_size > 10 * 1024 * 1024:
            print(f"[V8-CKPT] Найдены актуальные веса модели: {mt_file} ({mt_file.stat().st_size / (1024**2):.1f} МБ)")
            return c_dir

    for p in BASE_DIR.rglob("multitask_512_best.pth"):
        if p.is_file() and p.stat().st_size > 10 * 1024 * 1024:
            print(f"[V8-CKPT] Автоматически обнаружены веса модели: {p}")
            return p.parent

    print(f"[V8-CKPT WARN] Файл multitask_512_best.pth не найден, используется папка по умолчанию: {default_dir}")
    return default_dir


def _verify_cell2_synced() -> None:
    """
    Быстрый поведенческий тест (15 мс): гарантирует, что в памяти активны
    именно обновлённые функции из Ячейки 2 (проекции, фильтр глиом, дебайас /1.15, tau=0.80, Слой 4).
    """
    errors = []

    if abs(float(DEFAULT_SEG_THRESHOLD) - 0.80) > 1e-6:
        errors.append(f"DEFAULT_SEG_THRESHOLD={DEFAULT_SEG_THRESHOLD} (ожидалось 0.80).")

    ax_oval = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.ellipse(ax_oval, (256, 256), (145, 185), 0, 0, 360, 85, -1)
    p_ax = detect_slice_plane_and_level_normalized(ax_oval)
    if p_ax.get("plane") != "axial":
        errors.append(f"detect_slice_plane_and_level_normalized устарела (аксиальный овал -> {p_ax.get('plane')}).")

    if "_is_glioma_csf_or_ventricle_artifact" not in globals():
        errors.append("Отсутствует _is_glioma_csf_or_ventricle_artifact.")
    else:
        vault = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.ellipse(vault, (256, 256), (140, 175), 0, 0, 360, 255, -1)
        vb = vault > 0
        dv = cv2.distanceTransform(vault, cv2.DIST_L2, 3)
        img_g = np.full((IMG_SIZE, IMG_SIZE), 87.0, dtype=np.float32) * vb
        gli = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.circle(gli, (315, 165), 28, 255, -1)
        gli = cv2.bitwise_and(gli, vault)
        img_g[gli > 0] = 62.0
        if _is_glioma_csf_or_ventricle_artifact(gli > 0, img_g, dv, vb, 87.0, 10.0):
            errors.append("_is_glioma_csf_or_ventricle_artifact устарела (отбрасывает объёмную тёмную глиому).")

    dummy_vault = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.circle(dummy_vault, (256, 256), 140, 255, -1)
    dummy_gray = (dummy_vault > 0).astype(np.uint8) * 80
    dummy_p4 = np.zeros((4, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    dummy_p4[0] = 1.0
    pred_chk, _ = recalibrate_cls_probs_with_morphology(
        np.array([0.46, 0.49, 0.03, 0.02], dtype=np.float32), dummy_gray, dummy_vault, dummy_p4
    )
    if pred_chk != "glioma":
        errors.append("recalibrate_cls_probs_with_morphology устарела (не активна компенсация / 1.15).")

    if "compute_wt_uncertainty_map" not in globals():
        errors.append("В Ячейке 2 отсутствует compute_wt_uncertainty_map (Слой 4 неуверенности).")

    if errors:
        raise RuntimeError(
            "[CELL2-SYNC ERROR] Запуск остановлен — сначала выполните актуальную Ячейку 2:\n  - "
            + "\n  - ".join(errors)
        )
    print("[CELL2-SYNC OK] Ячейка 3 синхронизирована с Ячейкой 2 (tau=0.80, 2D-площадь, Слой 4 активен).")


def calibrate_and_refine_submasks_v7(
    parenchyma_gray_512: np.ndarray,
    det_probs_4ch: np.ndarray,
    parenchyma_mask_512: np.ndarray,
    predicted_class: str,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if predicted_class == "no_tumor":
        empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        return empty, empty, empty

    return calibrate_and_refine_submasks(
        parenchyma_gray_512=parenchyma_gray_512,
        det_probs_4ch=det_probs_4ch,
        parenchyma_mask_512=parenchyma_mask_512,
        predicted_class=predicted_class,
        seg_threshold=seg_threshold
    )


# ==============================================================================
# 2. ИНФЕРЕНС СРЕЗА СТРОГО ПО КОНВЕЙЕРУ ОБНОВЛЁННОЙ ЯЧЕЙКИ 2
# ==============================================================================
def _fast_audit_single_slice_v7(
    controller: StandaloneL4CDSSController,
    img_rgb_raw: np.ndarray,
    true_class: str,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD,
    unc_threshold: float = DEFAULT_UNC_THRESHOLD,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM
) -> Dict[str, Any]:
    rgb = cv2.cvtColor(img_rgb_raw, cv2.COLOR_GRAY2RGB) if img_rgb_raw.ndim == 2 else img_rgb_raw[:, :, :3]
    rgb_512 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    norm_gray = robust_percentile_normalize(rgb_512)
    plane_info = detect_slice_plane_and_level_normalized(norm_gray)
    is_vflipped = bool(plane_info.get("is_vflipped", False))
    plane = str(plane_info.get("plane", "axial"))
    silhouette_iou = float(plane_info.get("silhouette_iou_lr", 0.90))
    face_side = str(plane_info.get("face_side", "left"))

    canon_gray = cv2.flip(norm_gray, 0) if is_vflipped else norm_gray
    canon_vault = _compute_canonical_parenchyma_mask(canon_gray)
    orig_rgb = cv2.merge([norm_gray, norm_gray, norm_gray])
    canon_rgb = cv2.merge([canon_gray, canon_gray, canon_gray])

    damped_gray = np.where(canon_vault > 0, canon_gray, (canon_gray * 0.30).astype(np.uint8))
    damped_rgb = cv2.merge([damped_gray, damped_gray, damped_gray])

    t_can = controller.transforms_512(image=canon_rgb)["image"].unsqueeze(0).to(DEVICE)
    t_dmp = controller.transforms_512(image=damped_rgb)["image"].unsqueeze(0).to(DEVICE)
    batch_tta = torch.cat([t_can, torch.flip(t_can, dims=[3]), t_dmp, torch.flip(t_dmp, dims=[3])], dim=0).to(DEVICE)

    use_bf16 = bool(torch.cuda.is_available() and AMP_DTYPE == torch.bfloat16)
    with torch.no_grad():
        with amp_autocast(DEVICE, AMP_DTYPE):
            if getattr(controller, "gatekeeper_loaded", False):
                t_256 = controller.transforms_256(image=canon_rgb)["image"].unsqueeze(0).to(DEVICE)
                p_mri = float(F.softmax(controller.gatekeeper(t_256).float(), dim=-1)[0][1].cpu().item())
            else:
                p_mri = 1.0
            seg_logits_t, cls_logits_t, pooled_t = controller.model(batch_tta)

        probs_t = F.softmax(seg_logits_t.float(), dim=1)
        p0 = probs_t[0]
        p1 = torch.flip(probs_t[1], dims=[2])
        p2 = probs_t[2]
        p3 = torch.flip(probs_t[3], dims=[2])
        det_probs_4ch = (0.35 * p0 + 0.35 * p1 + 0.15 * p2 + 0.15 * p3).cpu().numpy()

        tta_wt_stack = torch.stack([
            torch.clamp(p[1] + p[2] + p[3], 0.0, 1.0) for p in (p0, p1, p2, p3)
        ], dim=0)
        tta_std_wt = torch.std(tta_wt_stack, dim=0, unbiased=False).cpu().numpy()

        cls_probs_all = F.softmax(cls_logits_t.float(), dim=-1).cpu().numpy()
        mean_cls_probs = 0.50 * cls_probs_all[0] + 0.50 * cls_probs_all[1]
        mean_emb = (0.5 * pooled_t[0] + 0.5 * pooled_t[1]).float().cpu().numpy()

    mahal_dist, mahal_ok = controller.mahal.verify_embedding(mean_emb)
    vault_bool_canon = (canon_vault > 127)
    p_wt_canon = np.clip(det_probs_4ch[1] + det_probs_4ch[2] + det_probs_4ch[3], 0.0, 1.0) * vault_bool_canon
    max_vault_p_wt = float(np.max(p_wt_canon)) if p_wt_canon.size > 0 else 0.0
    ood_passed = bool((p_mri >= 0.48) and (mahal_ok or max_vault_p_wt >= 0.55))

    pred_cls, recal_cls_probs, recal_meta = recalibrate_cls_probs_with_morphology_v7(
        mean_cls_probs=mean_cls_probs,
        gray_u8=canon_gray,
        vault_mask_u8=canon_vault,
        det_probs_4ch=det_probs_4ch
    )
    class_probs = {CLASS_NAMES[i]: float(recal_cls_probs[i]) for i in range(4)}

    active_seg_cls = pred_cls
    if pred_cls == "no_tumor":
        c_can = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        inf_can = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        e_can = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        unc_can = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        c_can, inf_can, e_can = calibrate_and_refine_submasks_v7(
            parenchyma_gray_512=canon_gray,
            det_probs_4ch=det_probs_4ch,
            parenchyma_mask_512=canon_vault,
            predicted_class=pred_cls,
            seg_threshold=seg_threshold
        )
        wt_can_u8 = (((c_can > 127) | (inf_can > 127) | (e_can > 127)).astype(np.uint8)) * 255
        mc_std_wt = controller._compute_vectorized_mc_dropout_std(t_can, n_passes=MC_DROPOUT_PASSES) if np.any(wt_can_u8 > 0) else None
        unc_can = compute_wt_uncertainty_map(
            det_probs_4ch=det_probs_4ch,
            wt_mask_u8=wt_can_u8,
            tta_std_wt=tta_std_wt,
            mc_std_wt=mc_std_wt
        )

    wt_can_bool = (c_can > 127) | (inf_can > 127) | (e_can > 127)
    dist_vault_canon = cv2.distanceTransform(canon_vault, cv2.DIST_L2, 3)
    blurred_canon = cv2.GaussianBlur(canon_gray, (5, 5), 1.0).astype(np.float32)
    if np.any(wt_can_bool):
        _, geom_quality = _evaluate_skull_and_orbit_artifact_penalty(
            wt_can_bool, blurred_canon, dist_vault_canon, vault_bool_canon, plane_info,
            is_meningioma=(active_seg_cls == "meningioma")
        )
    else:
        geom_quality = 0.0

    if is_vflipped:
        parenchyma_mask = cv2.flip(canon_vault, 0)
        mask_core = cv2.flip(c_can, 0)
        mask_inflam = cv2.flip(inf_can, 0)
        mask_edema = cv2.flip(e_can, 0)
        unc_map = cv2.flip(unc_can, 0)
    else:
        parenchyma_mask = canon_vault
        mask_core, mask_inflam, mask_edema = c_can, inf_can, e_can
        unc_map = unc_can

    morph = compute_three_zone_morphometry(
        mask_core, mask_inflam, mask_edema, parenchyma_mask,
        pixel_spacing_mm=pixel_spacing_mm, predicted_class=pred_cls, unc_map=unc_map,
        plane=plane
    )
    composite = render_three_zone_composite_512(
        orig_rgb, mask_core, mask_inflam, mask_edema, unc_map, unc_threshold=unc_threshold
    )

    composite = np.ascontiguousarray(composite, dtype=np.uint8).copy()
    v_cnts, _ = cv2.findContours(parenchyma_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if v_cnts:
        cv2.drawContours(composite, v_cnts, -1, (80, 230, 110), thickness=2, lineType=cv2.LINE_AA)

    brain_px = blurred_canon[vault_bool_canon & (blurred_canon >= 36.0) & (dist_vault_canon >= 8.0)]
    if brain_px.size < 200:
        brain_px = blurred_canon[vault_bool_canon & (blurred_canon >= 36.0)]
    if brain_px.size > 150:
        p15, p30, p48 = np.percentile(brain_px, [15.0, 30.0, 48.0])
        normal_med = float(0.50 * p30 + 0.50 * p48)
        normal_std = max(float((p48 - p15) / 1.04), 6.0)
        otsu_intra = float(np.atleast_1d(cv2.threshold(brain_px.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])[0])
    else:
        normal_med, normal_std, otsu_intra = 76.0, 10.0, 110.0

    mod_th = float(np.clip(normal_med + 1.15 * normal_std, 84.0, 134.0))
    hyper_th = float(np.clip(max(normal_med + 1.70 * normal_std, 0.84 * otsu_intra, normal_med + 15.0), 92.0, 144.0))

    vault_area = float(max(np.count_nonzero(vault_bool_canon), int(0.05 * IMG_SIZE * IMG_SIZE)))
    wt_px = int(np.count_nonzero(wt_can_bool))
    wt_vault_ratio = min(float(wt_px) / vault_area, 1.0)
    mean_tumor_int = float(np.mean(blurred_canon[wt_can_bool])) if wt_px > 0 else 0.0
    max_prob_in_tumor = float(np.max(p_wt_canon[wt_can_bool])) if wt_px > 0 else 0.0

    if pred_cls == "no_tumor":
        tier_label = "Zero-Mask (NoTumor)"
    elif wt_px == 0:
        tier_label = "MISSED (0 px)"
    elif pred_cls == "meningioma":
        if max_prob_in_tumor >= 0.24 and mean_tumor_int >= hyper_th:
            tier_label = "Tier-A (Seeded+Grow)"
        elif mean_tumor_int >= hyper_th:
            tier_label = "Tier-B (HyperNodule)"
        else:
            tier_label = "Tier-Neural (Pruned)"
    elif pred_cls == "glioma":
        tier_label = "Glioma-Neural" if max_prob_in_tumor >= seg_threshold * 0.85 else "Glioma-IntraCand"
    elif pred_cls == "pituitary":
        tier_label = "Pituitary-SellarV8"
    else:
        tier_label = "Standard-V8"

    if true_class == "unknown":
        cls_ok = True
        seg_warning = False
        error_reason = "OK"
        if pred_cls != "no_tumor":
            if wt_px < 45:
                seg_warning = True
                error_reason = "SEG_EMPTY (Пропуск узла)"
            elif wt_vault_ratio > 0.34:
                seg_warning = True
                error_reason = "SEG_OVERFLOW (>34% свода)"
        else:
            if wt_px > 0:
                seg_warning = True
                error_reason = "FP_ON_NORMAL (Ложное пятно)"
    else:
        cls_ok = (pred_cls == true_class)
        seg_warning = False
        error_reason = "OK"

        if not cls_ok:
            error_reason = f"CLS_ERR ({true_class[:4]}->{pred_cls[:4]})"
        if true_class != "no_tumor":
            if wt_px < 45:
                seg_warning = True
                error_reason = "SEG_EMPTY (Пропуск узла)" if cls_ok else f"{error_reason} + SEG_EMPTY"
            elif wt_vault_ratio > 0.34:
                seg_warning = True
                error_reason = "SEG_OVERFLOW (>34% свода)" if cls_ok else f"{error_reason} + SEG_OVERFLOW"
        else:
            if wt_px > 0:
                seg_warning = True
                error_reason = "FP_ON_NORMAL (Ложное пятно)" if cls_ok else f"{error_reason} + FP_ON_NORMAL"

    has_any_issue = (not cls_ok) or seg_warning

    return {
        "true_class": true_class,
        "predicted_class": pred_cls,
        "raw_pred_cls": recal_meta["raw_pred_cls"],
        "recalibrated": recal_meta["recalibrated"],
        "cls_ok": cls_ok,
        "seg_warning": seg_warning,
        "has_any_issue": has_any_issue,
        "error_reason": error_reason,
        "class_probs": class_probs,
        "morphometry": asdict(morph),
        "mahal_dist": float(mahal_dist),
        "ood_passed": ood_passed,
        "composite_with_vault_512": composite,
        "decision_trace": {
            "plane": plane,
            "is_vflipped": is_vflipped,
            "silhouette_iou_lr": round(silhouette_iou, 2),
            "face_side": face_side,
            "tier_label": tier_label,
            "geom_quality": round(geom_quality, 2),
            "intraaxial_score": recal_meta["intraaxial_score"],
            "has_ring_necrosis": int(recal_meta["has_ring_necrosis"]),
            "max_p_wt": round(max_vault_p_wt, 3),
            "mean_tumor_int": round(mean_tumor_int, 1),
            "normal_brain_med": round(normal_med, 1),
            "hyper_thresh": round(hyper_th, 1),
            "mod_thresh": round(mod_th, 1),
            "wt_px": wt_px,
            "wt_vault_pct": round(wt_vault_ratio * 100.0, 1),
            "vault_coverage_pct": round(100.0 * float(np.count_nonzero(vault_bool_canon)) / float(IMG_SIZE * IMG_SIZE), 1)
        }
    }


# ==============================================================================
# 3. ОТРИСОВКА КАРТОЧКИ НА 15 ФОТО (3x5) И СВОДНОГО ДАШБОРДА (ТОЛЬКО 2D-ПЛОЩАДЬ)
# ==============================================================================
def _render_15slice_card_to_file_v7(
    card_items: List[Dict[str, Any]],
    card_title_prefix: str,
    card_idx: int,
    total_cards: int,
    save_path: Path
) -> None:
    fig = plt.figure(figsize=(26, 21), dpi=150, facecolor="#0F172A")
    outer_gs = gridspec.GridSpec(
        3, 5, figure=fig, hspace=0.31, wspace=0.16,
        top=0.915, bottom=0.025, left=0.02, right=0.98
    )

    ok_cnt = sum(1 for x in card_items if not x["has_any_issue"])
    cls_ok_cnt = sum(1 for x in card_items if x["cls_ok"])
    recal_cnt = sum(1 for x in card_items if x.get("recalibrated", False))

    fig.suptitle(
        f"{card_title_prefix} — КАРТОЧКА #{card_idx:02d}/{total_cards:02d} "
        f"(Без нареканий: {ok_cnt}/{len(card_items)} | Точность класса: {cls_ok_cnt}/{len(card_items)} | Морф-калибровок: {recal_cnt})\n"
        f"Рамка: ЗЕЛЁНАЯ = норма | ОРАНЖЕВАЯ = сбой маски | КРАСНАЯ = ошибка класса   "
        f"[Ядро опухоли — красный | Воспаление / ГЭБ — оранжевый | Отёк — голубой | Неуверенная зона — пунктир]",
        fontsize=13.5, fontweight="bold", color="#F8FAFC", y=0.972
    )

    short_cls = ["Glioma", "Mening", "NoTum", "Pituit"]
    cls_keys = ["glioma", "meningioma", "no_tumor", "pituitary"]

    for slot_i in range(PHOTOS_PER_CARD):
        r_i, c_i = slot_i // 5, slot_i % 5
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[r_i, c_i], height_ratios=[3.05, 1.38], hspace=0.09
        )
        ax_img = fig.add_subplot(inner_gs[0])
        ax_bar = fig.add_subplot(inner_gs[1])

        if slot_i >= len(card_items):
            ax_img.axis("off")
            ax_bar.axis("off")
            continue

        item = card_items[slot_i]
        t_cls = item["true_class"]
        p_cls = item["predicted_class"]
        trace = item["decision_trace"]
        morph = item["morphometry"]
        probs = item["class_probs"]

        ax_img.imshow(item["composite_with_vault_512"])
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        if not item["cls_ok"]:
            border_col = "#EF4444"
        elif item["seg_warning"]:
            border_col = "#F59E0B"
        else:
            border_col = "#22C55E"

        for sp in ax_img.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(3.0 if item["has_any_issue"] else 2.0)

        conf_pct = probs.get(p_cls, 0.0) * 100.0
        vf_tag = " [VF-FLIP]" if trace["is_vflipped"] else ""
        recal_tag = f" [Recal:{item['raw_pred_cls'][:3]}->{p_cls[:3]}]" if item.get("recalibrated") else ""
        fname_short = item["filename"][:16]

        ax_img.set_title(
            f"#{item['global_idx']:03d} ({fname_short}) | {item['error_reason']}{vf_tag}\n"
            f"GT:{t_cls[:5]}->Pred:{p_cls[:5]} ({conf_pct:.0f}%){recal_tag} | {trace['plane'].upper()} | {trace['tier_label']}",
            fontsize=7.7, fontweight="bold", color=border_col, pad=3
        )

        info_box_txt = (
            f"WT:{morph['wt_area_mm2']:.0f}mm² ({trace['wt_vault_pct']:.1f}% свода) | "
            f"C:{morph['core_area_mm2']:.0f} Inf:{morph['inflam_area_mm2']:.0f} Ed:{morph['edema_area_mm2']:.0f} mm²\n"
            f"Int:{trace['mean_tumor_int']:.0f}/Med:{trace['normal_brain_med']:.0f} (T:{trace['mod_thresh']:.0f}/{trace['hyper_thresh']:.0f}) | "
            f"Unc:{morph['boundary_uncertainty_mean']:.2f} SilIoU:{trace['silhouette_iou_lr']:.2f}"
        )
        ax_img.text(
            0.02, 0.03, info_box_txt,
            transform=ax_img.transAxes, fontsize=6.7, color="#F8FAFC",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#0F172A", alpha=0.82, edgecolor="#334155")
        )

        ax_bar.set_facecolor("#1E293B")
        p_vals = [probs.get(k, 0.0) * 100.0 for k in cls_keys]
        b_cols = []
        for k in cls_keys:
            if k == p_cls and item["cls_ok"]:
                b_cols.append("#22C55E")
            elif k == p_cls and not item["cls_ok"]:
                b_cols.append("#EF4444")
            elif k == t_cls:
                b_cols.append("#F59E0B")
            else:
                b_cols.append("#64748B")

        y_pos = np.arange(4)
        bars = ax_bar.barh(y_pos, p_vals, color=b_cols, height=0.62, edgecolor="#0F172A")
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(short_cls, fontsize=7.3, color="#E2E8F0", fontweight="bold")
        ax_bar.set_xlim(0, 114)
        ax_bar.invert_yaxis()
        ax_bar.xaxis.set_visible(False)
        for sp in ax_bar.spines.values():
            sp.set_color("#334155")

        for b_rect, v_pct in zip(bars, p_vals):
            ax_bar.text(
                min(v_pct + 2.0, 95.0),
                b_rect.get_y() + b_rect.get_height() / 2.0,
                f"{v_pct:.1f}%", va="center", ha="left",
                fontsize=6.8, color="#F8FAFC", fontweight="bold"
            )

        ax_bar.set_title(
            f"P_WT:{trace['max_p_wt']:.2f} | IntraScore:{trace['intraaxial_score']:.2f} | Свод:{trace['vault_coverage_pct']:.0f}% | Mahal:{item['mahal_dist']:.1f}",
            fontsize=7.1, color="#94A3B8", pad=2
        )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _render_batch_summary_dashboard_v7(df_log: pd.DataFrame, target_cls: str, save_path: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(23, 13), dpi=160, facecolor="#F8FAFC")
    plt.subplots_adjust(hspace=0.34, wspace=0.26, top=0.89, bottom=0.06, left=0.05, right=0.97)

    acc = accuracy_score(df_log["true_class"], df_log["predicted_class"]) * 100.0
    clean_rate = (1.0 - df_log["has_any_issue"].mean()) * 100.0
    recal_count = int(df_log["recalibrated"].sum()) if "recalibrated" in df_log.columns else 0
    unique_gt = df_log["true_class"].nunique()

    fig.suptitle(
        f"АНАЛИТИКА МАССОВОГО ПРОГОНА V8 ({len(df_log)} СНИМКОВ | РЕЖИМ: {target_cls.upper()})\n"
        f"Общая точность классификации: {acc:.2f}% | Безупречных масок: {clean_rate:.2f}% | "
        f"Морф-калибровок: {recal_count}",
        fontsize=15.0, fontweight="bold", color="#0F172A", y=0.965
    )

    if unique_gt > 1:
        cm = confusion_matrix(df_log["true_class"], df_log["predicted_class"], labels=CLASS_NAMES)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0, 0], cbar=False
        )
        axes[0, 0].set_title("1. Матрица ошибок по 4 классам (GT vs Pred)", fontweight="bold")
        axes[0, 0].set_xlabel("Предсказанный класс")
        axes[0, 0].set_ylabel("Истинный класс (GT)")
    else:
        pred_counts = df_log["predicted_class"].value_counts().reindex(CLASS_NAMES, fill_value=0)
        bars1 = axes[0, 0].bar(CLASS_NAMES, pred_counts.values, color=["#3B82F6", "#10B981", "#64748B", "#8B5CF6"], edgecolor="black")
        axes[0, 0].set_title("1. Предсказанные классы на выборке", fontweight="bold")
        axes[0, 0].set_ylabel("Количество снимков")
        axes[0, 0].grid(axis="y", alpha=0.3)
        for b in bars1:
            axes[0, 0].annotate(f"{int(b.get_height())}", (b.get_x() + b.get_width() / 2, b.get_height()),
                                xytext=(0, 3), textcoords="offset points", ha="center", fontweight="bold")

    err_counts = df_log["error_reason"].value_counts().head(10)
    axes[0, 1].barh(err_counts.index[::-1], err_counts.values[::-1], color="#F97316", edgecolor="black")
    axes[0, 1].set_title("2. Диагностика качества (OK vs Типы ошибок)", fontweight="bold")
    axes[0, 1].set_xlabel("Количество снимков")
    axes[0, 1].grid(axis="x", alpha=0.3)

    tier_counts = df_log["tier_label"].value_counts()
    axes[0, 2].pie(
        tier_counts.values, labels=tier_counts.index, autopct="%1.1f%%",
        startangle=140, colors=sns.color_palette("Set2", len(tier_counts))
    )
    axes[0, 2].set_title("3. Эшелоны реконструкции узла (v8 Tiers)", fontweight="bold")

    valid_t = df_log[df_log["wt_px"] > 0]
    if len(valid_t) > 0:
        sc = axes[1, 0].scatter(
            valid_t["normal_brain_med"], valid_t["mean_tumor_int"],
            c=valid_t["max_p_wt"], cmap="plasma", alpha=0.8, edgecolors="k", s=42
        )
        plt.colorbar(sc, ax=axes[1, 0], label="Пик P_WT")
        x_l = np.linspace(max(20, valid_t["normal_brain_med"].min() - 5), min(180, valid_t["normal_brain_med"].max() + 5), 50)
        axes[1, 0].plot(x_l, x_l + 15.0, "r--", label="Порог контраста (+15 ед.)")
        axes[1, 0].legend(fontsize=8)
    axes[1, 0].set_title("4. Яркость опухоли vs Медиана мозга и P_WT", fontweight="bold")
    axes[1, 0].set_xlabel("Медиана нормальной паренхимы (ед. яркости)")
    axes[1, 0].set_ylabel("Средняя яркость выделенного узла")
    axes[1, 0].grid(alpha=0.3)

    if unique_gt > 1:
        cls_stats = []
        for c_name in CLASS_NAMES:
            sub_c = df_log[df_log["true_class"] == c_name]
            if len(sub_c) > 0:
                c_acc = sub_c["cls_ok"].mean() * 100.0
                c_clean = (1.0 - sub_c["has_any_issue"].mean()) * 100.0
                cls_stats.append((c_name, c_acc, c_clean, len(sub_c)))
        if cls_stats:
            x_idx = np.arange(len(cls_stats))
            w_b = 0.36
            b_acc = axes[1, 1].bar(x_idx - w_b/2, [s[1] for s in cls_stats], width=w_b, color="#22C55E", edgecolor="black", label="Точность класса (%)")
            b_cln = axes[1, 1].bar(x_idx + w_b/2, [s[2] for s in cls_stats], width=w_b, color="#3B82F6", edgecolor="black", label="Чистота маски (%)")
            axes[1, 1].set_xticks(x_idx)
            axes[1, 1].set_xticklabels([f"{s[0]}\n(n={s[3]})" for s in cls_stats], fontweight="bold")
            axes[1, 1].set_ylim(0, 112)
            axes[1, 1].set_title("5. Точность класса и чистота масок по 4 классам (%)", fontweight="bold")
            axes[1, 1].legend(fontsize=8)
            axes[1, 1].grid(axis="y", alpha=0.3)
            for bar_group in (b_acc, b_cln):
                for b in bar_group:
                    axes[1, 1].annotate(f"{b.get_height():.1f}%", (b.get_x() + b.get_width()/2, b.get_height()),
                                        xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7.5, fontweight="bold")
    else:
        axes[1, 1].hist(
            [df_log["core_area_mm2"], df_log["inflam_area_mm2"], df_log["edema_area_mm2"]],
            bins=20, stacked=True, color=["#E11D48", "#FF9E00", "#00BEFF"], edgecolor="black",
            label=["Ядро опухоли", "Воспаление / ГЭБ", "Отёк"]
        )
        axes[1, 1].set_title("5. Распределение 2D-площадей 3 субзон (мм²)", fontweight="bold")
        axes[1, 1].set_xlabel("Площадь (мм²)")
        axes[1, 1].set_ylabel("Число срезов")
        axes[1, 1].legend(fontsize=8)
        axes[1, 1].grid(alpha=0.3)

    df_plot = df_log.copy()
    df_plot["wt_vault_pct"] = np.clip(df_plot["wt_vault_pct"], 0.0, 100.0)
    x_col = "true_class" if unique_gt > 1 else "plane"
    sns.boxplot(data=df_plot, x=x_col, y="wt_vault_pct", hue=x_col, palette="Pastel1", ax=axes[1, 2], legend=False)
    axes[1, 2].axhline(34.0, color="red", linestyle="--", label="Верхний лимит (34% свода)")
    axes[1, 2].set_ylim(-2.0, max(45.0, float(df_plot["wt_vault_pct"].max()) + 6.0))
    axes[1, 2].set_title("6. Доля опухоли от площади свода черепа (%)", fontweight="bold")
    axes[1, 2].set_xlabel("Класс" if unique_gt > 1 else "Проекция среза")
    axes[1, 2].set_ylabel("Площадь WT / Площадь свода (%)")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(axis="y", alpha=0.3)

    fig.savefig(save_path, dpi=160)
    plt.close(fig)


# ==============================================================================
# 4. АВТООПРЕДЕЛЕНИЕ КЛАССА ФАЙЛА И СБОР СНИМКОВ
# ==============================================================================
def _infer_class_from_path_v8(filepath: Path, default_class: str = "glioma") -> str:
    parent_l = filepath.parent.name.lower()
    stem_l = filepath.stem.lower()
    full_str = f"{parent_l}_{stem_l}"
    clean_token = full_str.replace("_", "").replace("-", "").replace(" ", "")
    stem_up = filepath.stem.upper()

    if (
        any(k in clean_token for k in ("glioma", "глиом"))
        or stem_up.startswith(("G_", "TE-GL", "TR-GL", "TE_GL", "TR_GL", "GL_", "GL-"))
    ):
        return "glioma"
    if (
        any(k in clean_token for k in ("meningioma", "менингиом"))
        or stem_up.startswith(("M_", "TE-ME", "TR-ME", "TE_ME", "TR_ME", "ME_", "ME-"))
    ):
        return "meningioma"
    if (
        any(k in clean_token for k in ("pituitary", "гипофиз", "аденом"))
        or stem_up.startswith(("P_", "TE-PI", "TR-PI", "TE_PI", "TR_PI", "PI_", "PI-"))
    ):
        return "pituitary"
    if (
        any(k in clean_token for k in ("notumor", "healthy", "normal", "норма", "безопухол"))
        or stem_up.startswith(("N_", "TE-NO", "TR-NO", "TE_NO", "TR_NO", "NO_", "NO-"))
    ):
        return "no_tumor"

    if default_class == "unknown":
        return "unknown"
    return default_class if default_class in CLASS_NAMES else "glioma"


def _resolve_uploaded_or_local_images_v8(
    uploaded_files: Optional[List[Any]],
    source_mode: str,
    selected_class: str,
    max_slices_per_class: int
) -> List[Tuple[str, str]]:
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    target_classes = CLASS_NAMES if selected_class == "all_4_classes" else [selected_class]
    per_class_buckets: Dict[str, List[str]] = {c: [] for c in target_classes}
    seen_paths = set()

    def _try_add(fp_str: str, cls_name: str) -> None:
        if cls_name in per_class_buckets and fp_str not in seen_paths:
            if len(per_class_buckets[cls_name]) < int(max_slices_per_class):
                seen_paths.add(fp_str)
                per_class_buckets[cls_name].append(fp_str)

    if source_mode.startswith("📁") and uploaded_files:
        tmp_extract_dir = Path(tempfile.mkdtemp(prefix="mri_batch_v8_"))
        raw_files: List[Path] = []
        for f_obj in uploaded_files:
            f_path = Path(f_obj.name if hasattr(f_obj, "name") else str(f_obj))
            if not f_path.exists():
                continue
            if f_path.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(f_path, "r") as zf:
                        zf.extractall(tmp_extract_dir / f_path.stem)
                except Exception:
                    pass
            elif f_path.suffix.lower() in valid_exts:
                raw_files.append(f_path)

        for extr_p in sorted(tmp_extract_dir.rglob("*")):
            if extr_p.is_file() and extr_p.suffix.lower() in valid_exts:
                if "mask" not in extr_p.stem.lower() and "seg" not in extr_p.stem.lower():
                    raw_files.append(extr_p)

        for p in raw_files:
            c_name = _infer_class_from_path_v8(p, default_class=target_classes[0]) if selected_class == "all_4_classes" else selected_class
            _try_add(str(p.resolve()), c_name)
    else:
        manifest_csv = BASE_DIR / "neuro_cdss_512_data" / "manifests" / "mri_dataset_manifest.csv"
        if manifest_csv.exists():
            try:
                df_m = pd.read_csv(manifest_csv)
                for c_name in target_classes:
                    sub_df = df_m[df_m["class_name"] == c_name]
                    for fp in sub_df["filepath"].tolist():
                        if os.path.exists(str(fp)):
                            _try_add(str(fp), c_name)
            except Exception:
                pass

        if any(len(per_class_buckets[c]) < int(max_slices_per_class) for c in target_classes):
            search_dirs = [
                BASE_DIR / "neuro_cdss_512_data" / "mri_raw",
                Path.home() / ".cache" / "kagglehub" / "datasets",
                Path(tempfile.gettempdir())
            ]
            for s_dir in search_dirs:
                if not s_dir.exists():
                    continue
                for p in sorted(s_dir.rglob("*")):
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        if "mask" in p.stem.lower() or "seg" in p.stem.lower() or "card_" in p.stem.lower() or "dashboard" in p.stem.lower():
                            continue
                        c_name = _infer_class_from_path_v8(p, default_class="")
                        if c_name in per_class_buckets:
                            _try_add(str(p.resolve()), c_name)

        if sum(len(v) for v in per_class_buckets.values()) == 0:
            try:
                import kagglehub
                dl_root = Path(kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset"))
                for p in sorted(dl_root.rglob("*")):
                    if p.is_file() and p.suffix.lower() in valid_exts:
                        c_name = _infer_class_from_path_v8(p, default_class="")
                        if c_name in per_class_buckets:
                            _try_add(str(p.resolve()), c_name)
            except Exception as e:
                print(f"[V8-DATA WARN] Не удалось автоматически загрузить датасет с KaggleHub: {e}")

    final_items: List[Tuple[str, str]] = []
    for c_name in target_classes:
        for fp in per_class_buckets[c_name]:
            final_items.append((fp, c_name))
    return final_items


# ==============================================================================
# 5. ЯДРО МАССОВОГО ТЕСТИРОВАНИЯ
# ==============================================================================
def execute_mass_audit_v8(
    controller: StandaloneL4CDSSController,
    src_mode: str,
    target_cls: str,
    files_list: Optional[List[Any]],
    max_n_per_class: int,
    seg_t: float,
    err_first: bool,
    progress: Optional[Any] = None
) -> Tuple[str, Optional[str], Optional[str], List[str], str]:
    run_id = f"{int(time.time())}"
    run_dir = BATCH_CARDS_DIR / f"run_v8_{target_cls}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    items_to_process = _resolve_uploaded_or_local_images_v8(
        uploaded_files=files_list,
        source_mode=src_mode,
        selected_class=target_cls,
        max_slices_per_class=int(max_n_per_class)
    )
    if not items_to_process:
        return (
            "### ⚠️ Файлы не найдены! Загрузите фотографии или .zip архив в поле ниже.",
            None, None, [], ""
        )

    logs_list: List[Dict[str, Any]] = []
    per_class_results: Dict[str, List[Dict[str, Any]]] = {c: [] for c in CLASS_NAMES}
    error_buffer: List[Dict[str, Any]] = []
    regular_card_paths: List[str] = []
    t0 = time.time()

    iterator = progress.tqdm(items_to_process, desc=f"Массовый прогон v8 ({target_cls})") if progress is not None else items_to_process

    for i, (f_path, gt_cls) in enumerate(iterator):
        img_bgr = imread_unicode(f_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        res = _fast_audit_single_slice_v7(
            controller=controller,
            img_rgb_raw=img_rgb,
            true_class=gt_cls,
            seg_threshold=float(seg_t)
        )
        res["global_idx"] = i + 1
        res["filepath"] = f_path
        res["filename"] = Path(f_path).name

        trace = res["decision_trace"]
        morph = res["morphometry"]
        probs = res["class_probs"]

        logs_list.append({
            "global_idx": i + 1,
            "filename": res["filename"],
            "filepath": f_path,
            "true_class": gt_cls,
            "raw_pred_cls": res["raw_pred_cls"],
            "predicted_class": res["predicted_class"],
            "recalibrated": res["recalibrated"],
            "cls_ok": res["cls_ok"],
            "seg_warning": res["seg_warning"],
            "has_any_issue": res["has_any_issue"],
            "error_reason": res["error_reason"],
            "pred_confidence_pct": round(probs.get(res["predicted_class"], 0.0) * 100.0, 2),
            "plane": trace["plane"],
            "is_vflipped": trace["is_vflipped"],
            "silhouette_iou_lr": trace["silhouette_iou_lr"],
            "face_side": trace["face_side"],
            "tier_label": trace["tier_label"],
            "geom_quality": trace["geom_quality"],
            "intraaxial_score": trace["intraaxial_score"],
            "has_ring_necrosis": trace["has_ring_necrosis"],
            "max_p_wt": trace["max_p_wt"],
            "mean_tumor_int": trace["mean_tumor_int"],
            "normal_brain_med": trace["normal_brain_med"],
            "hyper_thresh": trace["hyper_thresh"],
            "mod_thresh": trace["mod_thresh"],
            "wt_px": trace["wt_px"],
            "wt_vault_pct": trace["wt_vault_pct"],
            "vault_coverage_pct": trace["vault_coverage_pct"],
            "wt_area_mm2": morph["wt_area_mm2"],
            "core_area_mm2": morph["core_area_mm2"],
            "inflam_area_mm2": morph["inflam_area_mm2"],
            "edema_area_mm2": morph["edema_area_mm2"],
            "boundary_uncertainty_mean": morph["boundary_uncertainty_mean"],
            "mahal_dist": round(res["mahal_dist"], 2)
        })

        per_class_results.setdefault(gt_cls, []).append(res)
        if res["has_any_issue"] and len(error_buffer) < 180:
            error_buffer.append(res)

    active_classes = [c for c in CLASS_NAMES if len(per_class_results.get(c, [])) > 0]
    for c_name in active_classes:
        c_items = per_class_results[c_name]
        tot_c_cards = math.ceil(len(c_items) / PHOTOS_PER_CARD)
        for c_idx in range(tot_c_cards):
            chunk = c_items[c_idx * PHOTOS_PER_CARD:(c_idx + 1) * PHOTOS_PER_CARD]
            c_path = run_dir / f"card_{c_name}_{c_idx + 1:02d}_of_{tot_c_cards:02d}.png"
            _render_15slice_card_to_file_v7(
                card_items=chunk,
                card_title_prefix=f"V8 КЛАСС: {c_name.upper()}",
                card_idx=c_idx + 1,
                total_cards=tot_c_cards,
                save_path=c_path
            )
            regular_card_paths.append(str(c_path))

    error_card_paths: List[str] = []
    if err_first and len(error_buffer) > 0:
        tot_err_cards = math.ceil(len(error_buffer) / PHOTOS_PER_CARD)
        for e_c_idx in range(tot_err_cards):
            chunk = error_buffer[e_c_idx * PHOTOS_PER_CARD:(e_c_idx + 1) * PHOTOS_PER_CARD]
            e_path = run_dir / f"00_ERROR_CARD_{e_c_idx + 1:02d}_of_{tot_err_cards:02d}_{target_cls}.png"
            _render_15slice_card_to_file_v7(
                card_items=chunk,
                card_title_prefix=f"⚠️ V8 СВОДКА ПРОБЛЕМНЫХ СРЕЗОВ ({target_cls.upper()})",
                card_idx=e_c_idx + 1,
                total_cards=tot_err_cards,
                save_path=e_path
            )
            error_card_paths.append(str(e_path))

    df_log = pd.DataFrame(logs_list)
    csv_p = run_dir / f"audit_log_v8_{target_cls}_{len(df_log)}slices.csv"
    df_log.to_csv(csv_p, index=False, encoding="utf-8")

    per_class_dashboard_paths: List[str] = []
    if len(active_classes) > 1:
        for c_name in active_classes:
            sub_df = df_log[df_log["true_class"] == c_name].reset_index(drop=True)
            if len(sub_df) > 0:
                sub_df.to_csv(run_dir / f"audit_log_v8_{c_name}_{len(sub_df)}slices.csv", index=False, encoding="utf-8")
                sub_dash_p = run_dir / f"01_dashboard_v8_{c_name}.png"
                _render_batch_summary_dashboard_v7(sub_df, c_name, sub_dash_p)
                per_class_dashboard_paths.append(str(sub_dash_p))

    dash_p = run_dir / f"00_analytics_dashboard_v8_{target_cls}.png"
    _render_batch_summary_dashboard_v7(df_log, target_cls, dash_p)

    zip_base = BATCH_CARDS_DIR / f"Batch_Audit_v8_{target_cls}_{len(df_log)}slices_{run_id}"
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(run_dir))

    elapsed = time.time() - t0
    n_tot = len(df_log)
    n_cls_ok = int(df_log["cls_ok"].sum())
    n_issues = int(df_log["has_any_issue"].sum())

    per_cls_lines = []
    for c_name in active_classes:
        sub_c = df_log[df_log["true_class"] == c_name]
        c_tot = len(sub_c)
        c_ok = int(sub_c["cls_ok"].sum())
        c_clean = int((~sub_c["has_any_issue"]).sum())
        per_cls_lines.append(
            f"  - **`{c_name}`**: проверено `{c_tot}` шт. | Точность класса: **`{c_ok}/{c_tot}` ({100.0*c_ok/max(c_tot,1):.2f}%)** | "
            f"Безупречных масок: **`{c_clean}/{c_tot}` ({100.0*c_clean/max(c_tot,1):.2f}%)**"
        )
    per_cls_md = "\n".join(per_cls_lines)

    summary_md = (
        f"### ✅ Массовый аудит v8 (`{target_cls}`) завершён за `{elapsed:.1f}` сек!\n"
        f"- **Всего проверено фотографий:** `{n_tot}` шт. (`{len(regular_card_paths)}` карточек по 15 фото)\n"
        f"- **Общая точность классификации:** `{n_cls_ok}/{n_tot}` (**`{100.0*n_cls_ok/max(n_tot,1):.2f}%`**)\n"
        f"- **Срезов с замечаниями (класс или маска):** `{n_issues}` шт. (`{100.0*(n_tot - n_issues)/max(n_tot,1):.2f}%` полностью чистых)\n"
        f"#### Поклассовая статистика:\n{per_cls_md}"
    )

    err_df = df_log[df_log["has_any_issue"]]
    if len(err_df) > 0:
        err_lines = [
            f"#{int(r.global_idx):03d} | [{r.true_class}->Pred:{r.predicted_class}] | {r.filename} | {r.error_reason} | "
            f"WT: {r.wt_area_mm2:.1f} мм² ({r.wt_vault_pct:.1f}% свода) | Tier: {r.tier_label}"
            for r in err_df.itertuples(index=False)
        ]
        err_text = "\n".join(err_lines)
    else:
        err_text = "Ошибок классификации и сегментации не обнаружено!"

    gallery_items = per_class_dashboard_paths + error_card_paths + regular_card_paths
    return summary_md, str(dash_p), str(zip_path), gallery_items, err_text




# ==============================================================================
# 5.5. ВИТРИНА-ПЛАШКА НА 20 МРТ ПО ПУБЛИЧНОЙ ССЫЛКЕ GOOGLE DRIVE (ВАРИАНТ 3)
# ==============================================================================
def _download_public_demo_20_images(
    public_url: str,
    file_urls: Optional[List[str]] = None,
    max_files: int = DEMO_20_MAX_FILES,
    force_redownload: bool = False
) -> Tuple[List[Tuple[str, str]], str]:
    """
    Скачивает до 20 МРТ-файлов по публичной ссылке Google Диска (папка, .zip-архив или список ссылок)
    без drive.mount() и возвращает список (filepath, inferred_true_class).
    """
    import gdown

    valid_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
    url_clean = (public_url or "").strip()
    extra_urls = [u.strip() for u in (file_urls or []) if isinstance(u, str) and u.strip()]

    if not url_clean and not extra_urls:
        return [], (
            "Укажите публичную ссылку Google Диска (на папку или .zip-архив с 20 МРТ) "
            "в Ячейке 1 (`PUBLIC_DEMO_20_GDRIVE_URL`) или в поле выше."
        )

    cache_key_src = f"{url_clean}|{'|'.join(extra_urls)}"
    cache_hash = hashlib.md5(cache_key_src.encode("utf-8")).hexdigest()[:12]
    dl_dir = DEMO_20_DIR / f"cache_{cache_hash}"

    if force_redownload and dl_dir.exists():
        shutil.rmtree(dl_dir, ignore_errors=True)
    dl_dir.mkdir(parents=True, exist_ok=True)

    def _collect_valid_imgs(root_dir: Path) -> List[Path]:
        imgs: List[Path] = []
        for p in sorted(root_dir.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() in valid_exts:
                low_stem = p.stem.lower()
                if any(bad in low_stem for bad in ("mask", "seg_", "card_", "dashboard", "plate_20")):
                    continue
                imgs.append(p)
        return imgs

    found_imgs = _collect_valid_imgs(dl_dir)

    if len(found_imgs) == 0:
        if url_clean:
            is_folder_url = ("/folders/" in url_clean) or ("drive.google.com/drive/u/" in url_clean and "folders" in url_clean)
            if is_folder_url:
                print(f"[DEMO-20] Скачивание публичной папки Google Drive: {url_clean}")
                gdown.download_folder(
                    url=url_clean,
                    output=str(dl_dir),
                    quiet=False,
                    use_cookies=False,
                    remaining_ok=True
                )
            else:
                print(f"[DEMO-20] Скачивание файла/архива по публичной ссылке Google Drive: {url_clean}")
                out_prefix = str(dl_dir) + os.sep
                if "http" in url_clean:
                    downloaded_p = gdown.download(url=url_clean, output=out_prefix, quiet=False, fuzzy=True)
                else:
                    downloaded_p = gdown.download(id=url_clean, output=out_prefix, quiet=False)

                if downloaded_p and os.path.exists(downloaded_p) and zipfile.is_zipfile(downloaded_p):
                    print(f"[DEMO-20] Распаковка ZIP-архива: {downloaded_p}")
                    with zipfile.ZipFile(downloaded_p, "r") as zf:
                        zf.extractall(dl_dir / "extracted_zip")

        if extra_urls:
            print(f"[DEMO-20] Скачивание {len(extra_urls)} отдельных файлов с Google Drive...")
            for idx_u, f_ref in enumerate(extra_urls[:int(max_files)], 1):
                try:
                    out_prefix = str(dl_dir) + os.sep
                    if "http" in f_ref:
                        dp = gdown.download(url=f_ref, output=out_prefix, quiet=False, fuzzy=True)
                    else:
                        dp = gdown.download(id=f_ref, output=out_prefix, quiet=False)
                    if dp and os.path.exists(dp) and zipfile.is_zipfile(dp):
                        with zipfile.ZipFile(dp, "r") as zf:
                            zf.extractall(dl_dir / f"extracted_{idx_u:02d}")
                except Exception as e:
                    print(f"[DEMO-20 WARN] Не удалось скачать файл #{idx_u}: {e}")

        # На случай если внутри скачанной папки оказался .zip-архив
        for z_path in sorted(dl_dir.rglob("*.zip")):
            try:
                if zipfile.is_zipfile(z_path):
                    with zipfile.ZipFile(z_path, "r") as zf:
                        zf.extractall(dl_dir / f"zip_{z_path.stem}")
            except Exception:
                pass

        found_imgs = _collect_valid_imgs(dl_dir)

    if not found_imgs:
        return [], (
            "Не удалось найти изображения МРТ по указанной ссылке. "
            "Проверьте, что для папки или .zip-архива открыт доступ «Все, у кого есть ссылка -> Читатель»."
        )

    selected = found_imgs[:int(max_files)]
    items = [(str(p.resolve()), _infer_class_from_path_v8(p, default_class="unknown")) for p in selected]
    return items, f"Загружено {len(items)} снимков МРТ из публичной ссылки Google Диска."


def _format_status_ru_demo20(item: Dict[str, Any]) -> str:
    if not item["cls_ok"]:
        return "Ошибка класса"
    if item["seg_warning"]:
        reason = item.get("error_reason", "")
        if "SEG_EMPTY" in reason:
            return "Замечание: пропуск узла"
        if "SEG_OVERFLOW" in reason:
            return "Замечание: >34% мозга"
        if "FP_ON_NORMAL" in reason:
            return "Замечание: ложное пятно"
        return "Замечание по маске"
    return "Хороший снимок"


def _render_20slice_showcase_plate(
    items: List[Dict[str, Any]],
    save_path: Path
) -> None:
    """
    Формирует единую плашку на все 20 выбранных МРТ (сетка 4 строки × 5 столбцов) с:
      - зелёной обводкой мозга (контур свода) на каждом снимке;
      - 3-зонной сегментацией и Слоем 4 (неуверенная зона пунктиром);
      - цветными рамками: зелёная = хороший снимок, оранжевая = замечание по маске, красная = неправильный случай;
      - русскоязычными подписями и столбцами вероятностей классов.
    """
    n_items = len(items)
    n_cols = 5
    n_rows = max(1, math.ceil(n_items / n_cols))

    fig_h = max(7.5, 6.8 * n_rows + 1.8)
    fig = plt.figure(figsize=(26, fig_h), dpi=150, facecolor="#0F172A")
    top_margin = 0.925 if n_rows >= 3 else 0.87
    outer_gs = gridspec.GridSpec(
        n_rows, n_cols, figure=fig, hspace=0.32, wspace=0.15,
        top=top_margin, bottom=0.02, left=0.018, right=0.982
    )

    good_cnt = sum(1 for x in items if (x["cls_ok"] and not x["seg_warning"]))
    orange_cnt = sum(1 for x in items if (x["cls_ok"] and x["seg_warning"]))
    bad_cnt = sum(1 for x in items if (not x["cls_ok"]))

    fig.suptitle(
        f"ДЕМОНСТРАЦИОННАЯ ПЛАШКА ({n_items} МРТ) — "
        f"Хороших: {good_cnt} | С оранжевой обводкой: {orange_cnt} | Неправильных: {bad_cnt}\n"
        f"Рамка: ЗЕЛЁНАЯ = хороший снимок | ОРАНЖЕВАЯ = замечание по маске | КРАСНАЯ = ошибка класса\n"
        f"На МРТ: зелёный контур — границы мозга | красный — ядро | оранжевый — воспаление / ГЭБ | голубой — отёк | пунктир — неуверенная зона",
        fontsize=13.5, fontweight="bold", color="#F8FAFC", y=0.978
    )

    short_cls_ru = ["Глиома", "Менингиома", "Норма", "Гипофиз"]
    cls_keys = ["glioma", "meningioma", "no_tumor", "pituitary"]

    total_slots = n_rows * n_cols
    for slot_i in range(total_slots):
        r_i, c_i = slot_i // n_cols, slot_i % n_cols
        inner_gs = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_gs[r_i, c_i], height_ratios=[3.15, 1.35], hspace=0.10
        )
        ax_img = fig.add_subplot(inner_gs[0])
        ax_bar = fig.add_subplot(inner_gs[1])

        if slot_i >= n_items:
            ax_img.axis("off")
            ax_bar.axis("off")
            continue

        item = items[slot_i]
        t_cls = item["true_class"]
        p_cls = item["predicted_class"]
        trace = item["decision_trace"]
        morph = item["morphometry"]
        probs = item["class_probs"]

        ax_img.imshow(item["composite_with_vault_512"])
        ax_img.set_xticks([])
        ax_img.set_yticks([])

        if not item["cls_ok"]:
            border_col = "#EF4444"   # Красная рамка — ошибка классификации
        elif item["seg_warning"]:
            border_col = "#F59E0B"   # Оранжевая рамка — замечание по маске
        else:
            border_col = "#22C55E"   # Зелёная рамка — хороший снимок

        for sp in ax_img.spines.values():
            sp.set_edgecolor(border_col)
            sp.set_linewidth(3.4 if item["has_any_issue"] else 2.4)

        conf_pct = probs.get(p_cls, 0.0) * 100.0
        fname_short = item["filename"][:18]
        status_ru = _format_status_ru_demo20(item)
        p_ru = CLASS_SHORT_RU.get(p_cls, p_cls)
        if t_cls in CLASS_NAMES:
            t_ru = CLASS_SHORT_RU.get(t_cls, t_cls)
            cls_line = f"Истина: {t_ru} → ИИ: {p_ru} ({conf_pct:.0f}%)"
        else:
            cls_line = f"ИИ: {p_ru} ({conf_pct:.0f}%)"

        ax_img.set_title(
            f"#{item['global_idx']:02d} ({fname_short}) | {status_ru}\n{cls_line}",
            fontsize=8.0, fontweight="bold", color=border_col, pad=3
        )

        info_box_txt = (
            f"Очаг: {morph['wt_area_mm2']:.0f} кв. мм ({trace['wt_vault_pct']:.1f}% мозга)\n"
            f"Ядро: {morph['core_area_mm2']:.0f} | Восп.: {morph['inflam_area_mm2']:.0f} | Отёк: {morph['edema_area_mm2']:.0f} | Неув.: {morph['boundary_uncertainty_mean']:.2f}"
        )
        ax_img.text(
            0.02, 0.03, info_box_txt,
            transform=ax_img.transAxes, fontsize=6.6, color="#F8FAFC",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#0F172A", alpha=0.84, edgecolor="#334155")
        )

        ax_bar.set_facecolor("#1E293B")
        p_vals = [probs.get(k, 0.0) * 100.0 for k in cls_keys]
        b_cols = []
        for k in cls_keys:
            if k == p_cls and item["cls_ok"]:
                b_cols.append("#22C55E")
            elif k == p_cls and not item["cls_ok"]:
                b_cols.append("#EF4444")
            elif k == t_cls:
                b_cols.append("#F59E0B")
            else:
                b_cols.append("#64748B")

        y_pos = np.arange(4)
        bars = ax_bar.barh(y_pos, p_vals, color=b_cols, height=0.62, edgecolor="#0F172A")
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(short_cls_ru, fontsize=7.2, color="#E2E8F0", fontweight="bold")
        ax_bar.set_xlim(0, 118)
        ax_bar.invert_yaxis()
        ax_bar.xaxis.set_visible(False)
        for sp in ax_bar.spines.values():
            sp.set_color("#334155")

        for b_rect, v_pct in zip(bars, p_vals):
            ax_bar.text(
                min(v_pct + 2.0, 96.0),
                b_rect.get_y() + b_rect.get_height() / 2.0,
                f"{v_pct:.1f}%", va="center", ha="left",
                fontsize=6.8, color="#F8FAFC", fontweight="bold"
            )

    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def execute_demo_20_showcase(
    controller: StandaloneL4CDSSController,
    public_url: str,
    seg_t: float = DEFAULT_SEG_THRESHOLD,
    unc_t: float = DEFAULT_UNC_THRESHOLD,
    force_redownload: bool = False,
    file_urls: Optional[List[str]] = None,
    progress: Optional[Any] = None
) -> Tuple[str, Optional[str], Optional[str], str]:
    """
    Загружает 20 заранее выбранных МРТ по публичной ссылке Google Диска,
    прогоняет их через модель с зелёной обводкой мозга и собирает единую плашку на все 20 файлов.
    """
    items_paths, dl_msg = _download_public_demo_20_images(
        public_url=public_url,
        file_urls=file_urls,
        max_files=DEMO_20_MAX_FILES,
        force_redownload=bool(force_redownload)
    )
    if not items_paths:
        return f"### ⚠️ {dl_msg}", None, None, ""

    t0 = time.time()
    processed_items: List[Dict[str, Any]] = []
    iterator = progress.tqdm(items_paths, desc="Анализ 20 МРТ для плашки") if progress is not None else items_paths

    for idx_i, (f_path, gt_cls) in enumerate(iterator, 1):
        img_bgr = imread_unicode(f_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        res = _fast_audit_single_slice_v7(
            controller=controller,
            img_rgb_raw=img_rgb,
            true_class=gt_cls,
            seg_threshold=float(seg_t),
            unc_threshold=float(unc_t)
        )
        res["global_idx"] = idx_i
        res["filepath"] = f_path
        res["filename"] = Path(f_path).name
        processed_items.append(res)

    if not processed_items:
        return "### ⚠️ Не удалось прочитать ни одного изображения из загруженной папки.", None, None, ""

    run_id = int(time.time())
    plate_path = DEMO_20_DIR / f"plate_20_mri_{run_id}.png"
    _render_20slice_showcase_plate(processed_items, plate_path)

    elapsed = time.time() - t0
    n_tot = len(processed_items)
    good_cnt = sum(1 for x in processed_items if (x["cls_ok"] and not x["seg_warning"]))
    orange_cnt = sum(1 for x in processed_items if (x["cls_ok"] and x["seg_warning"]))
    bad_cnt = sum(1 for x in processed_items if (not x["cls_ok"]))
    labeled_items = [x for x in processed_items if x["true_class"] in CLASS_NAMES]

    acc_line = ""
    if labeled_items:
        acc_val = 100.0 * sum(1 for x in labeled_items if x["cls_ok"]) / len(labeled_items)
        acc_line = f"- **Точность классификации по размеченным снимкам:** `{sum(1 for x in labeled_items if x['cls_ok'])}/{len(labeled_items)}` (**`{acc_val:.1f}%`**)\n"

    summary_md = (
        f"### ✅ Плашка на `{n_tot}` МРТ готова за `{elapsed:.1f}` сек!\n"
        f"- **Зелёная рамка (хорошие снимки):** `{good_cnt}` шт.\n"
        f"- **Оранжевая рамка (замечания по маске):** `{orange_cnt}` шт.\n"
        f"- **Красная рамка (неправильные случаи / ошибка класса):** `{bad_cnt}` шт.\n"
        f"{acc_line}"
        f"- *На каждом снимке нанесена зелёная обводка границ мозга, 3-зонная маска и пунктир неуверенной зоны.*"
    )

    detail_lines = []
    for x in processed_items:
        st_ru = _format_status_ru_demo20(x)
        t_ru = CLASS_SHORT_RU.get(x["true_class"], x["true_class"])
        p_ru = CLASS_SHORT_RU.get(x["predicted_class"], x["predicted_class"])
        conf = x["class_probs"].get(x["predicted_class"], 0.0) * 100.0
        m = x["morphometry"]
        tr = x["decision_trace"]
        gt_part = f"Истина: {t_ru} -> " if x["true_class"] in CLASS_NAMES else ""
        detail_lines.append(
            f"#{x['global_idx']:02d} | {x['filename']} | {st_ru} | {gt_part}ИИ: {p_ru} ({conf:.1f}%) | "
            f"Очаг: {m['wt_area_mm2']:.1f} кв. мм ({tr['wt_vault_pct']:.1f}% мозга) | Неуверенность: {m['boundary_uncertainty_mean']:.3f}"
        )

    return summary_md, str(plate_path), str(plate_path), "\n".join(detail_lines)


# ==============================================================================
# 6. ИНТЕРФЕЙС GRADIO (ВКЛАДКА 1: ПЛАШКА 20 МРТ | ВКЛАДКА 2: 1 СРЕЗ | ВКЛАДКА 3: МАССОВЫЙ АУДИТ)
# ==============================================================================
def launch_interactive_cdss_with_batch_auditor_v8() -> None:
    _verify_cell2_synced()
    best_ckpt_dir = _resolve_best_checkpoint_dir(CKPT_DIR)
    controller = StandaloneL4CDSSController(best_ckpt_dir)

    # Автоматическая сборка плашки на 20 МРТ при запуске, если ссылка указана в Ячейке 1
    init_demo20_md = (
        "### Готово к загрузке 20 МРТ с Google Диска\n"
        "Вставьте публичную ссылку на папку или `.zip`-архив с 20 МРТ-снимками (либо задайте `PUBLIC_DEMO_20_GDRIVE_URL` в Ячейке 1) "
        "и нажмите кнопку ниже."
    )
    init_demo20_plate = None
    init_demo20_file = None
    init_demo20_details = ""

    has_demo20_cfg = bool((PUBLIC_DEMO_20_GDRIVE_URL or "").strip() or len(PUBLIC_DEMO_20_GDRIVE_FILES or []) > 0)
    if AUTO_RUN_DEMO_20_ON_STARTUP and has_demo20_cfg:
        try:
            print("[DEMO-20] Автозапуск загрузки 20 МРТ с Google Диска и сборки единой плашки...")
            init_demo20_md, init_demo20_plate, init_demo20_file, init_demo20_details = execute_demo_20_showcase(
                controller=controller,
                public_url=PUBLIC_DEMO_20_GDRIVE_URL,
                seg_t=DEFAULT_SEG_THRESHOLD,
                unc_t=DEFAULT_UNC_THRESHOLD,
                force_redownload=False,
                file_urls=PUBLIC_DEMO_20_GDRIVE_FILES,
                progress=None
            )
        except Exception as exc_init:
            tb_init = traceback.format_exc()
            print(tb_init)
            init_demo20_md = f"### ⚠️ Ошибка автозагрузки 20 МРТ:\n```\n{exc_init}\n```"
            init_demo20_details = tb_init

    init_summary_md = "### Готово к запуску массового тестирования..."
    init_dash_path = None
    init_zip_path = None
    init_gallery: List[str] = []
    init_err_text = ""

    with gr.Blocks(title="СППР МРТ головного мозга — Витрина 20 МРТ, Клинический анализ и Аудит") as demo:
        gr.Markdown(
            "## 🧠 Нейроонкологическая СППР МРТ головного мозга\n"
            f"> *{AI_DISCLAIMER_HEADER}*"
        )

        with gr.Tabs():
            # ==================================================================
            # ВКЛАДКА 1 (ОТКРЫВАЕТСЯ ПЕРВОЙ): ПЛАШКА НА 20 МРТ С GOOGLE ДИСКА
            # ==================================================================
            with gr.TabItem("📋 Демо-плашка 20 МРТ (Google Диск)"):
                gr.Markdown(
                    "### Единая плашка на 20 заранее выбранных МРТ по публичной ссылке Google Диска\n"
                    "На каждом снимке отображается **зелёная обводка границ мозга**, 3-зонная сегментация, пунктир неуверенной зоны "
                    "и цветная рамка: **зелёная** — хороший снимок, **оранжевая** — замечание по маске, **красная** — неправильный случай (ошибка класса)."
                )
                with gr.Row():
                    with gr.Column(scale=7):
                        demo20_url_box = gr.Textbox(
                            value=(PUBLIC_DEMO_20_GDRIVE_URL or "").strip(),
                            label="Публичная ссылка Google Диска на папку или .zip-архив с 20 файлами МРТ",
                            placeholder="https://drive.google.com/drive/folders/... или ссылка на .zip-архив",
                            lines=1
                        )
                        demo20_adv_state = gr.State(False)
                        demo20_adv_btn = gr.Button("⚙️ Продвинутые настройки", variant="secondary")
                        with gr.Group(visible=False) as demo20_adv_group:
                            demo20_seg_slider = gr.Slider(
                                minimum=0.20, maximum=0.90, value=DEFAULT_SEG_THRESHOLD, step=0.05,
                                label="Порог выделения опухоли",
                                info=(
                                    "Насколько строго модель выделяет границы очага. "
                                    "Влево — область выделения станет шире (захватит слабые и размытые края, но могут появиться лишние участки). "
                                    "Вправо — выделение станет строже и меньше (останутся только самые явные участки, мелкие края могут обрезаться)."
                                )
                            )
                            demo20_unc_slider = gr.Slider(
                                minimum=0.15, maximum=0.75, value=DEFAULT_UNC_THRESHOLD, step=0.02,
                                label="Порог неуверенной зоны",
                                info=(
                                    "Управляет пунктирной обводкой спорных участков внутри выделения. "
                                    "Влево — пунктиром отметится большая область (даже при небольших сомнениях модели). "
                                    "Вправо — пунктир останется только в самых спорных местах или исчезнет, если модель уверена."
                                )
                            )
                        with gr.Row():
                            demo20_force_chk = gr.Checkbox(
                                value=False,
                                label="Скачать заново с Google Диска (сбросить кэш)"
                            )
                            demo20_run_btn = gr.Button(
                                "🚀 Загрузить 20 МРТ и собрать плашку",
                                variant="primary"
                            )
                    with gr.Column(scale=5):
                        demo20_status_md = gr.Markdown(value=init_demo20_md)
                        demo20_download = gr.File(
                            value=init_demo20_file,
                            label="Скачать готовую плашку на 20 МРТ в высоком разрешении (.png)"
                        )

                demo20_plate_preview = gr.Image(
                    value=init_demo20_plate,
                    type="filepath",
                    label="Единая плашка на 20 МРТ (зелёный контур — границы мозга | рамки: зелёная / оранжевая / красная)",
                    height=920
                )
                demo20_details_box = gr.Textbox(
                    value=init_demo20_details,
                    label="Подробный разбор всех 20 снимков",
                    lines=10
                )

                def on_run_demo20(url_str, seg_t, unc_t, force_dl, progress=gr.Progress(track_tqdm=True)):
                    try:
                        return execute_demo_20_showcase(
                            controller=controller,
                            public_url=url_str,
                            seg_t=float(seg_t),
                            unc_t=float(unc_t),
                            force_redownload=bool(force_dl),
                            file_urls=PUBLIC_DEMO_20_GDRIVE_FILES if not (url_str or "").strip() else None,
                            progress=progress
                        )
                    except Exception as exc:
                        tb = traceback.format_exc()
                        return f"### ❌ Ошибка:\n```\n{exc}\n```", None, None, tb

                def _toggle_demo20_adv(is_open):
                    new_open = not bool(is_open)
                    btn_lbl = "🔼 Скрыть продвинутые настройки" if new_open else "⚙️ Продвинутые настройки"
                    return new_open, gr.update(visible=new_open), gr.update(value=btn_lbl)

                demo20_adv_btn.click(
                    fn=_toggle_demo20_adv,
                    inputs=[demo20_adv_state],
                    outputs=[demo20_adv_state, demo20_adv_group, demo20_adv_btn]
                )
                demo20_run_btn.click(
                    fn=on_run_demo20,
                    inputs=[demo20_url_box, demo20_seg_slider, demo20_unc_slider, demo20_force_chk],
                    outputs=[demo20_status_md, demo20_plate_preview, demo20_download, demo20_details_box]
                )

            # ==================================================================
            # ВКЛАДКА 2: КЛИНИЧЕСКИЙ РАЗБОР 1 СРЕЗА + ХОЛСТ С МАСКОЙ + DOCX
            # ==================================================================
            with gr.TabItem("🏥 Клинический разбор 1 среза + Интерактивная кисть (Приказ № 359н)"):
                session_state = gr.State({})
                with gr.Row():
                    with gr.Column(scale=6):
                        img_input = gr.Image(type="numpy", label="Исходный срез МРТ", height=300)
                        result_preview = gr.Image(
                            type="numpy",
                            label="Результат выделения (Ядро опухоли — красный | Воспаление / ГЭБ — оранжевый | Отёк — голубой | Неуверенная зона — пунктир)",
                            height=300
                        )
                        analyze_btn = gr.Button("Запустить анализ МРТ", variant="primary")
                        canvas_editor = gr.ImageEditor(
                            type="numpy",
                            canvas_size=(IMG_SIZE, IMG_SIZE),
                            label="Холст ручной коррекции (Ластик стирает маску модели и ручные мазки | Кисти: Красный — Ядро, Оранжевый — Воспаление / ГЭБ, Голубой — Отёк, Чёрный — Ластик)",
                            brush=gr.Brush(colors=["#E11D48", "#FF9E00", "#00BEFF", "#000000"], default_size=8),
                            eraser=gr.Eraser(default_size=14),
                            layers=True,
                            height=450
                        )
                        apply_brush_btn = gr.Button("Применить правки кистью/ластиком и пересчитать площадь")
                        single_adv_state = gr.State(False)
                        single_adv_btn = gr.Button("⚙️ Продвинутые настройки", variant="secondary")
                        with gr.Group(visible=False) as single_adv_group:
                            seg_slider = gr.Slider(
                                minimum=0.20, maximum=0.90, value=DEFAULT_SEG_THRESHOLD, step=0.05,
                                label="Порог выделения опухоли",
                                info=(
                                    "Насколько строго модель выделяет границы очага. "
                                    "Влево — область выделения станет шире (захватит слабые и размытые края, но могут появиться лишние участки). "
                                    "Вправо — выделение станет строже и меньше (останутся только самые явные участки, мелкие края могут обрезаться)."
                                )
                            )
                            unc_slider = gr.Slider(
                                minimum=0.15, maximum=0.75, value=DEFAULT_UNC_THRESHOLD, step=0.02,
                                label="Порог неуверенной зоны",
                                info=(
                                    "Управляет пунктирной обводкой спорных участков внутри выделения. "
                                    "Влево — пунктиром отметится большая область (даже при небольших сомнениях модели). "
                                    "Вправо — пунктир останется только в самых спорных местах или исчезнет, если модель уверена."
                                )
                            )
                            spacing_slider = gr.Slider(
                                minimum=0.40, maximum=1.50, value=DEFAULT_PIXEL_SPACING_MM, step=0.0125,
                                label="Масштаб пикселя (мм)",
                                info=(
                                    "Физический размер одного пикселя для расчёта площади в кв. мм (сам контур на снимке не меняет). "
                                    "Влево — рассчитанная площадь и размеры в отчёте уменьшатся. "
                                    "Вправо — площадь и размеры увеличатся."
                                )
                            )

                    with gr.Column(scale=6):
                        telemetry_md = gr.Markdown("### Готово к работе — загрузите срез МРТ")
                        protocol_box = gr.Textbox(label="Протокол описания МРТ головного мозга", lines=22)
                        with gr.Row():
                            export_btn = gr.Button("Скачать протокол в Word (.docx)", variant="primary")
                            docx_file = gr.File(label="Готовый файл протокола")

                def on_analyze(img, seg_t, unc_t, px_sp, state):
                    if img is None:
                        return state, None, None, "### Загрузите МРТ-срез", ""
                    res = controller.predict_and_quantify(
                        img, seg_threshold=float(seg_t), unc_threshold=float(unc_t), pixel_spacing_mm=float(px_sp)
                    )
                    if not res.get("ood_passed", False):
                        editor_val = build_canvas_editor_payload(res["composite_512"])
                        return state, res["composite_512"], editor_val, "### 🛑 Изображение не распознано как МРТ мозга", res["protocol_text"]
                    editor_val = build_canvas_editor_payload(
                        res["orig_rgb_512"], res["mask_core_512"], res["mask_inflam_512"], res["mask_edema_512"]
                    )
                    state.update(res)
                    m = res["morphometry"]
                    cls_ru = CLASS_LABELS_RU.get(res["predicted_class"], res["predicted_class"])
                    tel = (
                        f"### Предварительный результат ИИ: {cls_ru}\n"
                        f"- **Ядро опухоли:** `{m['core_area_mm2']}` кв. мм\n"
                        f"- **Воспаление / ГЭБ:** `{m['inflam_area_mm2']}` кв. мм\n"
                        f"- **Отёк:** `{m['edema_area_mm2']}` кв. мм\n"
                        f"- **Общая площадь очага:** `{m['wt_area_mm2']}` кв. мм (`{m['max_diameter_mm']} × {m['perp_diameter_mm']}` мм)\n"
                        f"- **Средняя неуверенность границы:** `{m['boundary_uncertainty_mean']}` *(пунктир — спорная область)*"
                    )
                    return state, res["composite_512"], editor_val, tel, res["protocol_text"]

                def on_brush(editor_data, seg_t, unc_t, px_sp, proto, state):
                    if not state or "orig_rgb_512" not in state:
                        return state, None, None, "Нет данных", proto
                    t0 = time.time()
                    c, inf, e = apply_brush_strokes_to_submasks(
                        editor_data, state["mask_core_512"], state["mask_inflam_512"], state["mask_edema_512"]
                    )
                    morph = compute_three_zone_morphometry(
                        c, inf, e, state["parenchyma_mask_512"],
                        pixel_spacing_mm=float(px_sp), predicted_class=state["predicted_class"],
                        unc_map=state["unc_map_512"], plane=state.get("plane", "axial")
                    )
                    comp = render_three_zone_composite_512(
                        state["orig_rgb_512"], c, inf, e, state["unc_map_512"], unc_threshold=float(unc_t)
                    )
                    cpu_ms = (time.time() - t0) * 1000.0
                    state["mask_core_512"], state["mask_inflam_512"], state["mask_edema_512"] = c, inf, e
                    state["composite_512"] = comp
                    state["morphometry"] = asdict(morph)
                    editor_val = build_canvas_editor_payload(state["orig_rgb_512"], c, inf, e)
                    synced_proto = sync_protocol_preserving_doctor_edits(
                        proto, state["predicted_class"],
                        state["class_probs"][state["predicted_class"]] * 100.0,
                        morph, state["ood_metrics"]["mahalanobis_distance"], True
                    )
                    tel = (
                        f"Пересчёт за {cpu_ms:.1f} мс | Ядро опухоли: `{morph.core_area_mm2}` кв. мм | "
                        f"Воспаление / ГЭБ: `{morph.inflam_area_mm2}` кв. мм | Отёк: `{morph.edema_area_mm2}` кв. мм | "
                        f"Общая площадь: `{morph.wt_area_mm2}` кв. мм"
                    )
                    return state, comp, editor_val, tel, synced_proto

                def _toggle_single_adv(is_open):
                    new_open = not bool(is_open)
                    btn_lbl = "🔼 Скрыть продвинутые настройки" if new_open else "⚙️ Продвинутые настройки"
                    return new_open, gr.update(visible=new_open), gr.update(value=btn_lbl)

                single_adv_btn.click(
                    fn=_toggle_single_adv,
                    inputs=[single_adv_state],
                    outputs=[single_adv_state, single_adv_group, single_adv_btn]
                )
                analyze_btn.click(
                    fn=on_analyze,
                    inputs=[img_input, seg_slider, unc_slider, spacing_slider, session_state],
                    outputs=[session_state, result_preview, canvas_editor, telemetry_md, protocol_box]
                )
                apply_brush_btn.click(
                    fn=on_brush,
                    inputs=[canvas_editor, seg_slider, unc_slider, spacing_slider, protocol_box, session_state],
                    outputs=[session_state, result_preview, canvas_editor, telemetry_md, protocol_box]
                )
                export_btn.click(
                    fn=lambda proto, st: export_protocol_to_docx(
                        proto,
                        st.get("composite_512", np.zeros((512, 512, 3), np.uint8)),
                        st.get("morphometry", {}),
                        st.get("predicted_class", "no_tumor")
                    ),
                    inputs=[protocol_box, session_state],
                    outputs=[docx_file]
                )

            # ==================================================================
            # ВКЛАДКА 3: МАССОВЫЙ ФОРЕНЗИК-АУДИТ ПО 4 КЛАССАМ / СВОИМ ФАЙЛАМ
            # ==================================================================
            with gr.TabItem("🔬 Массовый Форензик-Аудит v8 (Карточки по 15 фото)"):
                gr.Markdown(
                    "### Пакетный прогон по своим файлам / `.zip` архиву с матрицей ошибок, дашбордом и карточками по 15 снимков"
                )
                with gr.Row():
                    with gr.Column(scale=5):
                        source_radio = gr.Radio(
                            choices=[
                                "📁 Загрузить свои фото или .zip архив с папками классов с компьютера",
                                "⚡ Взять автоматически из тестовой выборки в среде Colab по выбранному классу"
                            ],
                            value="📁 Загрузить свои фото или .zip архив с папками классов с компьютера",
                            label="1. Источник МРТ-снимков"
                        )
                        class_selector = gr.Radio(
                            choices=["all_4_classes", "glioma", "meningioma", "pituitary", "no_tumor"],
                            value="all_4_classes",
                            label="2. Режим тестирования (`all_4_classes` = все 4 класса сразу)"
                        )
                        batch_files = gr.File(
                            file_count="multiple",
                            label="3. Загрузите фотографии или .zip архив с компьютера",
                            height=180
                        )
                        max_slices_slider = gr.Slider(
                            minimum=15, maximum=900, value=300, step=15,
                            label="Максимум снимков на один класс",
                            info=(
                                "Сколько фотографий каждого класса проверить за один запуск. "
                                "Влево — проверка пройдёт быстрее. "
                                "Вправо — проверится больше снимков и статистика будет полнее, но времени потребуется больше."
                            )
                        )
                        batch_adv_state = gr.State(False)
                        batch_adv_btn = gr.Button("⚙️ Продвинутые настройки", variant="secondary")
                        with gr.Group(visible=False) as batch_adv_group:
                            seg_tau_batch = gr.Slider(
                                minimum=0.20, maximum=0.90, value=DEFAULT_SEG_THRESHOLD, step=0.05,
                                label="Порог выделения опухоли",
                                info=(
                                    "Насколько строго модель выделяет границы очага. "
                                    "Влево — область выделения станет шире (захватит слабые края, но возможны лишние участки). "
                                    "Вправо — выделение станет строже и меньше (мелкие или тусклые очаги могут обрезаться)."
                                )
                            )
                        error_first_chk = gr.Checkbox(
                            value=True,
                            label="Поставить первыми в галерее поклассовые дашборды и «Карточки ТОЛЬКО ОШИБОК»"
                        )
                        run_batch_btn = gr.Button(
                            "🚀 Запустить массовое тестирование",
                            variant="primary"
                        )

                    with gr.Column(scale=7):
                        batch_status_md = gr.Markdown(value=init_summary_md)
                        dashboard_preview = gr.Image(
                            value=init_dash_path,
                            type="filepath",
                            label="Сводный дашборд массового прогона (2D-площадь)",
                            height=420
                        )
                        with gr.Row():
                            zip_download = gr.File(
                                value=init_zip_path,
                                label="Скачать ZIP-архив со всеми карточками (по 15 фото), дашбордами и CSV-логами"
                            )

                gr.Markdown("### 🖼 Галерея дашбордов и карточек по 15 фотографий")
                cards_gallery = gr.Gallery(
                    value=init_gallery,
                    label="Карточки сегментации по 15 фото + поклассовые дашборды",
                    columns=2,
                    height=780,
                    object_fit="contain"
                )
                error_files_box = gr.Textbox(
                    value=init_err_text,
                    label="Детальный список проблемных файлов по результатам массового прогона",
                    lines=10
                )

                def on_run_batch_audit_v8(
                    src_mode: str,
                    target_cls: str,
                    files_list: Optional[List[Any]],
                    max_n: int,
                    seg_t: float,
                    err_first: bool,
                    progress=gr.Progress(track_tqdm=True)
                ):
                    try:
                        return execute_mass_audit_v8(
                            controller=controller,
                            src_mode=src_mode,
                            target_cls=target_cls,
                            files_list=files_list,
                            max_n_per_class=int(max_n),
                            seg_t=float(seg_t),
                            err_first=bool(err_first),
                            progress=progress
                        )
                    except Exception as exc:
                        tb = traceback.format_exc()
                        return f"### ❌ Ошибка:\n```\n{exc}\n```", None, None, [], tb

                def _toggle_batch_adv(is_open):
                    new_open = not bool(is_open)
                    btn_lbl = "🔼 Скрыть продвинутые настройки" if new_open else "⚙️ Продвинутые настройки"
                    return new_open, gr.update(visible=new_open), gr.update(value=btn_lbl)

                batch_adv_btn.click(
                    fn=_toggle_batch_adv,
                    inputs=[batch_adv_state],
                    outputs=[batch_adv_state, batch_adv_group, batch_adv_btn]
                )
                run_batch_btn.click(
                    fn=on_run_batch_audit_v8,
                    inputs=[source_radio, class_selector, batch_files, max_slices_slider, seg_tau_batch, error_first_chk],
                    outputs=[batch_status_md, dashboard_preview, zip_download, cards_gallery, error_files_box]
                )

    demo.queue().launch(
        server_name=os.environ.get("CDSS_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("CDSS_PORT", "7860")),
        share=os.environ.get("CDSS_SHARE", "0") == "1",
        inbrowser=os.environ.get("CDSS_NO_BROWSER", "0") != "1",
        debug=False,
    )


if __name__ == "__main__":
    launch_interactive_cdss_with_batch_auditor_v8()
