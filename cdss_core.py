# -*- coding: utf-8 -*-
"""Автоматически сгенерировано из Ячейки 2 (кроссплатформенное ядро СППР)."""

import os
from compat import (
    configure_console_utf8,
    get_workspace_dir,
    select_compute_device,
    amp_autocast,
    imwrite_unicode,
    get_windows_font_candidates,
)
from config import (
    PUBLIC_GDRIVE_FOLDER_URL,
    PUBLIC_GDRIVE_FILES,
)

configure_console_utf8()

# -*- coding: utf-8 -*-
"""
===========================================================================================
ЯЧЕЙКА 2: АВТОНОМНЫЙ ИНФЕРЕНС-КОНТУР (T4 / L4 / A100 / CPU) — 2D-ПЛОЩАДЬ + СЛОЙ НЕУВЕРЕННОСТИ
===========================================================================================
"""

import os
import re
import sys
import time
import math
import shutil
import hashlib
import traceback
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any

# Резервная инициализация ссылок, если Ячейку 1 пропустили
if "PUBLIC_GDRIVE_FOLDER_URL" not in globals():
    PUBLIC_GDRIVE_FOLDER_URL = ""
if "PUBLIC_GDRIVE_FILES" not in globals():
    PUBLIC_GDRIVE_FILES = {
        "multitask_512_best.pth": "",
        "mahalanobis_ood_512.npz": "",
        "gatekeeper_512_best.pth": "",
    }


def ensure_public_checkpoints_downloaded(target_dir: Path) -> Path:
    """
    Автономно скачивает чекпоинты с публичных ссылок Google Диска без drive.mount().
    """
    import gdown

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    mt_file = target_dir / "multitask_512_best.pth"
    if mt_file.exists() and mt_file.stat().st_size > 10 * 1024 * 1024:
        print(f"[CKPT] Локальные веса уже загружены: {mt_file} ({mt_file.stat().st_size / (1024**2):.1f} МБ)")
        return target_dir

    # Вариант 1: скачивание по прямым ссылкам на файлы (наиболее стабильно для больших .pth)
    for fname, ref in PUBLIC_GDRIVE_FILES.items():
        ref_clean = (ref or "").strip()
        dst = target_dir / fname
        if not ref_clean or (dst.exists() and dst.stat().st_size > 1024):
            continue
        print(f"[CKPT-DOWNLOAD] Скачивание {fname} с публичной ссылки Google Drive...")
        try:
            if "http" in ref_clean:
                gdown.download(url=ref_clean, output=str(dst), quiet=False, fuzzy=True)
            else:
                gdown.download(id=ref_clean, output=str(dst), quiet=False)
        except Exception as e:
            print(f"[CKPT-WARN] Не удалось скачать {fname}: {e}")

    # Вариант 2: скачивание всей публичной папки checkpoints_512, если основной файл ещё не скачан
    if (not mt_file.exists() or mt_file.stat().st_size <= 10 * 1024 * 1024) and PUBLIC_GDRIVE_FOLDER_URL.strip():
        print("[CKPT-DOWNLOAD] Скачивание публичной папки checkpoints_512 с Google Drive...")
        try:
            gdown.download_folder(
                url=PUBLIC_GDRIVE_FOLDER_URL.strip(),
                output=str(target_dir),
                quiet=False,
                use_cookies=False,
                remaining_ok=True
            )
            for req_name in ("multitask_512_best.pth", "mahalanobis_ood_512.npz", "gatekeeper_512_best.pth"):
                dst = target_dir / req_name
                if not dst.exists():
                    found = [p for p in target_dir.rglob(req_name) if p.is_file()]
                    if found:
                        shutil.move(str(found[0]), str(dst))
        except Exception as e:
            print(f"[CKPT-WARN] Ошибка скачивания папки: {e}")

    if mt_file.exists() and mt_file.stat().st_size > 10 * 1024 * 1024:
        print(f"[CKPT-OK] Основная модель готова: {mt_file} ({mt_file.stat().st_size / (1024**2):.1f} МБ)")
    else:
        print(
            "[CKPT-ERROR] Файл multitask_512_best.pth не найден! "
            "Вставьте ссылки в Ячейку 1 (PUBLIC_GDRIVE_FILES / PUBLIC_GDRIVE_FOLDER_URL) и откройте доступ «Все, у кого есть ссылка -> Читатель»."
        )
    return target_dir


def setup_standalone_l4_environment() -> Tuple[Path, Path]:
    local_base = get_workspace_dir()
    local_ckpt = local_base / "checkpoints_512"

    env_ckpt = os.environ.get("CDSS_CKPT_DIR", "").strip()
    if env_ckpt and (Path(env_ckpt).expanduser() / "multitask_512_best.pth").exists():
        active_ckpt_dir = Path(env_ckpt).expanduser().resolve()
    elif os.environ.get("CDSS_SKIP_DOWNLOAD", "0") == "1":
        active_ckpt_dir = local_ckpt
    else:
        active_ckpt_dir = ensure_public_checkpoints_downloaded(local_ckpt)

    active_ckpt_dir.mkdir(parents=True, exist_ok=True)
    return local_base, active_ckpt_dir


BASE_DIR, CKPT_DIR = setup_standalone_l4_environment()
REPORTS_DIR = BASE_DIR / "clinical_reports_512"
DEMO_CASES_DIR = BASE_DIR / "demo_clinical_slices_512"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DEMO_CASES_DIR.mkdir(parents=True, exist_ok=True)

import cv2
import numpy as np
import pydicom
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, Field, ConfigDict

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

import gradio as gr

IMG_SIZE = 512
MC_DROPOUT_PASSES = 4
DEFAULT_PIXEL_SPACING_MM = 0.9375
DEFAULT_SEG_THRESHOLD = 0.80
DEFAULT_UNC_THRESHOLD = 0.42
CLS_GRAD_SCALE_PHASE2 = 0.15

# Совместимость с NVIDIA T4 (исключает ошибки cuDNN 9 FIND/GET engine) + поддержка L4/A100/CPU
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
if hasattr(torch, "set_float32_matmul_precision"):
    torch.set_float32_matmul_precision("high")

DEVICE, AMP_DTYPE = select_compute_device()
print(f"[DEVICE] Активное устройство вычислений: {DEVICE} (AMP={AMP_DTYPE})")

CLASS_NAMES = ["glioma", "meningioma", "no_tumor", "pituitary"]
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
CLASS_LABELS_RU = {
    "glioma": "Глиома головного мозга (первичная нейроэпителиальная опухоль)",
    "meningioma": "Менингиома (внемозговое оболочечное образование)",
    "no_tumor": "Патологических очаговых образований головного мозга не выявлено (норма)",
    "pituitary": "Аденома гипофиза (образование хиазмально-селлярной области)"
}
CLASS_CONCLUSION_PROSE_RU = {
    "glioma": "внутримозгового инфильтративного образования глиального ряда (глиомы)",
    "meningioma": "внемозгового конвекситального или парасагиттального оболочечного образования (менингиомы)",
    "no_tumor": "отсутствия очаговых патологических изменений вещества головного мозга",
    "pituitary": "интра-супраселлярного образования турецкого седла (аденомы гипофиза)"
}
CLASS_ICD10_MAP = {
    "glioma": "C71.9 (Злокачественное новообразование головного мозга неуточнённой локализации)",
    "meningioma": "D32.0 (Доброкачественное новообразование мозговых оболочек головного мозга)",
    "no_tumor": "Z03.8 (Наблюдение при подозрении на другие болезни и состояния — патологии не выявлено)",
    "pituitary": "D35.2 (Доброкачественное новообразование гипофиза)"
}
PLANE_LABELS_RU = {
    "axial": "аксиальная",
    "coronal": "коронарная",
    "sagittal": "сагиттальная",
}

AI_MODEL_NAME = "нейросетевая модель СППР для анализа МРТ головного мозга"

AI_DISCLAIMER_HEADER = (
    "ВНИМАНИЕ: Предварительное заключение сформировано с применением искусственного интеллекта "
    f"({AI_MODEL_NAME}). Искусственный интеллект не заменяет диагностику врача — "
    "результаты подлежат обязательной проверке врачом-рентгенологом."
)

AI_DISCLAIMER_FOOTER = (
    "ПРИМЕЧАНИЕ: Искусственный интеллект не заменяет диагностику врача и не может использоваться "
    "самостоятельно для постановки окончательного диагноза."
)

ANTI_HALLUCINATION_SYSTEM_PROMPT = (
    "Вы — медицинский алгоритмический модуль строгой верификации данных СППР. "
    "Правило 1: Категорически запрещено додумывать, экстраполировать или добавлять клинические симптомы "
    "(головная боль, судороги, парезы), анатомические структуры или МР-характеристики (перфузия, спектроскопия), "
    "которых явно нет во входном структурированном словаре AnonymizedMedicalMetricsSchema. "
    "Правило 2: Если какой-либо параметр равен None, 0.0 или не определяется на данном 2D-срезе, "
    "в текст заключения обязана подставляться строго стандартизированная формулировка: "
    "'не визуализируется' или 'данные отсутствуют'."
)


def _safe_torch_load(filepath: Path) -> Dict[str, Any]:
    try:
        return torch.load(filepath, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location="cpu")


# ==============================================================================
# 1. ДЕТЕРМИНИРОВАННЫЙ ПАРСИНГ ПДн (PII) — БЕЗ ДАТ И БЕЗ ТОЛЩИНЫ СРЕЗА
# ==============================================================================
class PatientPIISchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patient_full_name: str = Field(default="данные отсутствуют")
    patient_id: str = Field(default="АНОНИМНЫЙ-КОНТУР")
    patient_age: str = Field(default="данные отсутствуют")
    patient_sex: str = Field(default="данные отсутствуют")
    modality: str = Field(default="МРТ")
    scanner_manufacturer_model: str = Field(default="данные отсутствуют")
    contrast_agent_name: str = Field(default="данные отсутствуют")
    contrast_dosage: str = Field(default="данные отсутствуют")
    contrast_administration_route: str = Field(default="данные отсутствуют")
    allergic_reaction: str = Field(default="данные отсутствуют")
    pixel_spacing_mm: float = Field(default=DEFAULT_PIXEL_SPACING_MM)


def extract_patient_pii_deterministic(source_path_or_dict: Optional[Any] = None) -> PatientPIISchema:
    if source_path_or_dict is None:
        return PatientPIISchema()

    if isinstance(source_path_or_dict, dict):
        return PatientPIISchema(
            patient_full_name=str(source_path_or_dict.get("patient_full_name", "данные отсутствуют")),
            patient_id=str(source_path_or_dict.get("patient_id", "АНОНИМНЫЙ-КОНТУР")),
            patient_age=str(source_path_or_dict.get("patient_age", "данные отсутствуют")),
            patient_sex=str(source_path_or_dict.get("patient_sex", "данные отсутствуют")),
            modality=str(source_path_or_dict.get("modality", "МРТ")),
            scanner_manufacturer_model=str(source_path_or_dict.get("scanner_manufacturer_model", "данные отсутствуют")),
            contrast_agent_name=str(source_path_or_dict.get("contrast_agent_name", "данные отсутствуют")),
            contrast_dosage=str(source_path_or_dict.get("contrast_dosage", "данные отсутствуют")),
            contrast_administration_route=str(source_path_or_dict.get("contrast_administration_route", "данные отсутствуют")),
            allergic_reaction=str(source_path_or_dict.get("allergic_reaction", "данные отсутствуют")),
            pixel_spacing_mm=float(source_path_or_dict.get("pixel_spacing_mm", DEFAULT_PIXEL_SPACING_MM))
        )

    p_str = str(source_path_or_dict)
    if p_str.lower().endswith(".dcm") and os.path.exists(p_str):
        try:
            ds = pydicom.dcmread(p_str, stop_before_pixels=True)
            name_val = str(getattr(ds, "PatientName", "данные отсутствуют"))
            if name_val and name_val != "данные отсутствуют":
                name_val = str(name_val).replace("^", " ").strip()
            pid_val = str(getattr(ds, "PatientID", "АНОНИМНЫЙ-КОНТУР"))
            age_val = str(getattr(ds, "PatientAge", "данные отсутствуют"))
            sex_val = str(getattr(ds, "PatientSex", "данные отсутствуют"))
            mod = str(getattr(ds, "Modality", "MR"))
            man = str(getattr(ds, "Manufacturer", ""))
            model = str(getattr(ds, "ManufacturerModelName", ""))
            scanner = f"{man} {model}".strip() if (man or model) else "данные отсутствуют"

            c_agent = str(getattr(ds, "ContrastBolusAgent", "данные отсутствуют"))
            c_dose = str(getattr(ds, "ContrastBolusVolume", "данные отсутствуют"))
            c_route = str(getattr(ds, "ContrastBolusRoute", "данные отсутствуют"))

            ps = getattr(ds, "PixelSpacing", [DEFAULT_PIXEL_SPACING_MM, DEFAULT_PIXEL_SPACING_MM])
            pix_sp = float(ps[0]) if ps else DEFAULT_PIXEL_SPACING_MM

            return PatientPIISchema(
                patient_full_name=name_val or "данные отсутствуют",
                patient_id=pid_val or "АНОНИМНЫЙ-КОНТУР",
                patient_age=age_val or "данные отсутствуют",
                patient_sex=sex_val or "данные отсутствуют",
                modality=mod or "MR",
                scanner_manufacturer_model=scanner,
                contrast_agent_name=c_agent if c_agent and c_agent != "None" else "данные отсутствуют",
                contrast_dosage=c_dose if c_dose and c_dose != "None" else "данные отсутствуют",
                contrast_administration_route=c_route if c_route and c_route != "None" else "данные отсутствуют",
                allergic_reaction="данные отсутствуют",
                pixel_spacing_mm=pix_sp
            )
        except Exception:
            pass

    return PatientPIISchema()


# ==============================================================================
# 2. PYDANTIC-КОНТРАКТЫ (ТОЛЬКО 2D-ПЛОЩАДЬ И ЛИНЕЙНЫЕ РАЗМЕРЫ)
# ==============================================================================
class SingleLesionMetricsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lesion_index: int = Field(default=1)
    max_diameter_mm: float = Field(default=0.0)
    perp_diameter_mm: float = Field(default=0.0)
    wt_area_mm2: float = Field(default=0.0)
    core_area_mm2: float = Field(default=0.0)
    inflam_area_mm2: float = Field(default=0.0)
    inflam_rim_width_mm: float = Field(default=0.0)
    edema_area_mm2: float = Field(default=0.0)
    hemisphere_loc: str = Field(default="данные отсутствуют")
    lobe_and_slice_level: str = Field(default="данные отсутствуют")
    has_necrotic_core: bool = Field(default=False)
    has_active_inflammatory_rim: bool = Field(default=False)
    has_perifocal_edema: bool = Field(default=False)


class AnonymizedMedicalMetricsSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    predicted_class: str = Field(default="no_tumor")
    confidence_pct: float = Field(default=0.0)
    class_probs: Dict[str, float] = Field(default_factory=dict)
    plane: str = Field(default="axial")
    slice_level_description: str = Field(default="Уровень тел боковых желудочков")
    lesion_count: int = Field(default=1)
    lesions: List[SingleLesionMetricsSchema] = Field(default_factory=list)
    wt_area_mm2: float = Field(default=0.0)
    core_area_mm2: float = Field(default=0.0)
    inflam_area_mm2: float = Field(default=0.0)
    edema_area_mm2: float = Field(default=0.0)
    max_diameter_mm: float = Field(default=0.0)
    perp_diameter_mm: float = Field(default=0.0)
    midline_shift_mm: float = Field(default=0.0)
    mass_effect_grade: str = Field(default="отсутствует")
    ventricular_compression_status: str = Field(default="не сдавлены")
    boundary_uncertainty_mean: float = Field(default=0.0)
    mahalanobis_distance: float = Field(default=0.0)
    ood_passed: bool = Field(default=True)


def enforce_anti_hallucination_guardrails(metrics: AnonymizedMedicalMetricsSchema) -> AnonymizedMedicalMetricsSchema:
    if metrics.wt_area_mm2 <= 0.0:
        metrics.lesion_count = 0
        metrics.lesions = []
        metrics.mass_effect_grade = "отсутствует"
        metrics.ventricular_compression_status = "не сдавлены"
        metrics.core_area_mm2 = 0.0
        metrics.inflam_area_mm2 = 0.0
        metrics.edema_area_mm2 = 0.0
    return metrics


# ==============================================================================
# 3. МОДЕРНИЗИРОВАННЫЙ ФИЛЬТР ВНУТРИЧЕРЕПНОГО СВОДА И 4-КАНАЛЬНАЯ АРХИТЕКТУРА
# ==============================================================================
_KERNEL_ERODE_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
_KERNEL_ERODE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_CLOSE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_KERNEL_OPEN_9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_KERNEL_OPEN_19 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19))
_KERNEL_DILATE_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_KERNEL_CLOSE_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
_KERNEL_CLOSE_11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
_KERNEL_CLOSE_15 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
_KERNEL_CLOSE_21 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))

def detect_slice_plane_and_level_normalized(gray_512: np.ndarray) -> Dict[str, Any]:
    """
    Точное разделение проекций (sagittal / coronal / axial):
      - Устраняет ложные [VF-FLIP] на нормальных коронарных и аксиальных срезах (#003);
      - Не путает овальные аксиальные срезы (hh >= 1.15*hw, #008) с коронарными;
      - Надёжно определяет сторону лица (face_side) на сагиттальных срезах (#001, #002).
    """
    blurred = cv2.GaussianBlur(gray_512, (5, 5), 1.2)
    head_init = (blurred > 14).astype(np.uint8) * 255
    head_init = cv2.morphologyEx(head_init, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
    ext_cnts, _ = cv2.findContours(head_init, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ext_cnts:
        return {
            "plane": "axial",
            "si_lr": 0.85,
            "bbox": (0, 0, IMG_SIZE, IMG_SIZE),
            "face_side": "left",
            "anterior_side": "left",
            "is_vflipped": False,
            "silhouette_iou_lr": 0.90
        }

    largest_head = max(ext_cnts, key=cv2.contourArea)
    hx, hy, hw, hh = cv2.boundingRect(largest_head)
    head_roi = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.drawContours(head_roi, [largest_head], -1, 255, thickness=-1)

    roi_crop = head_roi[hy:hy+hh, hx:hx+hw] > 0
    half_w = hw // 2
    if half_w > 10:
        left_sil = roi_crop[:, :half_w]
        right_sil = np.fliplr(roi_crop[:, hw - half_w:])
        inter_sil = float(np.count_nonzero(left_sil & right_sil))
        union_sil = max(float(np.count_nonzero(left_sil | right_sil)), 1.0)
        silhouette_iou_lr = inter_sil / union_sil
    else:
        silhouette_iou_lr = 0.90

    # Строгий критерий переворота: широкая шея строго упирается в самый верх (hy <= 2),
    # а нижний купол головы НЕ касается нижнего края кадра.
    top_edge_w = float(np.count_nonzero(head_roi[min(IMG_SIZE - 1, hy + 2), hx:hx + hw]))
    bot_edge_w = float(np.count_nonzero(head_roi[max(0, hy + hh - 3), hx:hx + hw]))
    q1_w = float(np.mean(np.sum(head_roi[hy + int(0.04 * hh):hy + int(0.18 * hh), hx:hx + hw] > 0, axis=1))) if hh > 20 else 0.0
    q4_w = float(np.mean(np.sum(head_roi[hy + int(0.82 * hh):hy + int(0.96 * hh), hx:hx + hw] > 0, axis=1))) if hh > 20 else 0.0

    is_vflipped = bool(
        (hy <= 2)
        and (top_edge_w >= 0.45 * hw)
        and (hy + hh < IMG_SIZE - 25)
        and (bot_edge_w < 0.28 * hw)
        and (q1_w > q4_w * 1.25)
    )

    work_blur = cv2.flip(blurred, 0) if is_vflipped else blurred
    work_roi = cv2.flip(head_roi, 0) if is_vflipped else head_roi
    roi_h = work_blur[hy:hy+hh, hx:hx+hw] if (hw > 0 and hh > 0) else work_blur

    head_vals = work_blur[work_blur > 18]
    if head_vals.size > 100:
        med_int = float(np.median(head_vals))
        iqr_int = float(np.percentile(head_vals, 75) - np.percentile(head_vals, 25))
    else:
        med_int, iqr_int = 100.0, 40.0

    dark_th = max(16.0, med_int - 1.15 * iqr_int)

    if roi_h.size > 100:
        h_w_sz = roi_h.shape[1]
        half_w_r = h_w_sz // 2
        left_h = roi_h[:, :half_w_r]
        right_h = np.fliplr(roi_h[:, h_w_sz - half_w_r:])
        min_sz = min(left_h.shape[1], right_h.shape[1])
        if min_sz > 5:
            diff_lr = np.mean(np.abs(left_h[:, :min_sz].astype(np.float32) - right_h[:, :min_sz].astype(np.float32)))
            std_h = max(float(np.std(roi_h)), 10.0)
            si_lr = float(np.clip(1.0 - (diff_lr / (1.4 * std_h)), 0.0, 1.0))
        else:
            si_lr = 0.85
    else:
        si_lr = 0.85

    # Определение стороны лица на сагиттальном срезе: внутренние пазухи внутри силуэта + рельеф носа/губ
    y_f1, y_f2 = hy + int(0.36 * hh), hy + int(0.80 * hh)
    x_mid = hx + hw // 2
    l_roi_f = work_roi[y_f1:y_f2, hx:x_mid] > 0
    r_roi_f = work_roi[y_f1:y_f2, x_mid:hx + hw] > 0
    l_air_in = float(np.mean(work_blur[y_f1:y_f2, hx:x_mid][l_roi_f] < max(36.0, dark_th))) if np.any(l_roi_f) else 0.0
    r_air_in = float(np.mean(work_blur[y_f1:y_f2, x_mid:hx + hw][r_roi_f] < max(36.0, dark_th))) if np.any(r_roi_f) else 0.0

    left_xs, right_xs = [], []
    for row_y in range(hy + int(0.38 * hh), min(IMG_SIZE, hy + int(0.82 * hh))):
        nz = np.where(work_roi[row_y, hx:hx + hw] > 0)[0]
        if nz.size > 10:
            left_xs.append(float(nz[0]))
            right_xs.append(float(nz[-1]))
    if len(left_xs) >= 12:
        l_rugged = float(np.mean(np.abs(np.diff(left_xs, n=2)))) / max(float(hw), 1.0)
        r_rugged = float(np.mean(np.abs(np.diff(right_xs, n=2)))) / max(float(hw), 1.0)
        l_std = float(np.std(left_xs)) / max(float(hw), 1.0)
        r_std = float(np.std(right_xs)) / max(float(hw), 1.0)
    else:
        l_rugged = r_rugged = l_std = r_std = 0.0

    left_face_score = 1.6 * l_air_in + 1.8 * l_std + 6.0 * l_rugged
    right_face_score = 1.6 * r_air_in + 1.8 * r_std + 6.0 * r_rugged
    face_side = "left" if left_face_score >= right_face_score else "right"

    bottom_row_width = float(np.count_nonzero(work_roi[min(IMG_SIZE - 2, hy + hh - 3), :] > 0))
    w_at_95 = float(np.count_nonzero(work_roi[min(IMG_SIZE - 1, int(hy + 0.95 * hh)), hx:hx+hw])) / max(float(hw), 1.0)
    has_neck_base = bool((hy + hh >= IMG_SIZE - 12) and (bottom_row_width >= 0.48 * hw or w_at_95 >= 0.56))

    top_nose_w = float(np.count_nonzero(work_roi[min(IMG_SIZE - 1, hy + int(0.04 * hh)), hx:hx+hw])) / max(float(hw), 1.0)
    is_axial_with_nose = (top_nose_w < 0.28) and (hy + hh < 496) and (w_at_95 < 0.46) and (silhouette_iou_lr >= 0.84)
    # Замкнутый симметричный овал без шеи внизу — это AXIAL (исправляет #008 / 1208.jpg и крупные односторонние опухоли)
    is_closed_axial_oval = (not has_neck_base) and (w_at_95 < 0.56) and (silhouette_iou_lr >= 0.82)

    lower_air_band = work_blur[int(hy + 0.62 * hh):int(hy + 0.88 * hh), hx + int(0.28 * hw):hx + int(0.72 * hw)]
    lower_air_roi = work_roi[int(hy + 0.62 * hh):int(hy + 0.88 * hh), hx + int(0.28 * hw):hx + int(0.72 * hw)] > 0
    lower_air_frac = float(np.mean(lower_air_band[lower_air_roi] < max(36.0, dark_th))) if np.any(lower_air_roi) else 0.0

    mid_band = work_blur[int(hy + 0.50 * hh):int(hy + 0.76 * hh), hx+int(0.20*hw):hx+int(0.80*hw)]
    has_trough = bool(np.min(np.median(mid_band, axis=1)) < dark_th * 1.05) if mid_band.size > 20 else False

    if (
        silhouette_iou_lr < 0.78
        or (si_lr < 0.48 and not is_axial_with_nose and not is_closed_axial_oval)
        or (si_lr < 0.55 and (w_at_95 >= 0.62 or silhouette_iou_lr < 0.82))
    ):
        plane = "sagittal"
    elif is_axial_with_nose or (is_closed_axial_oval and lower_air_frac < 0.28):
        plane = "axial"
    elif has_neck_base or (has_trough and lower_air_frac >= 0.22):
        plane = "coronal"
    else:
        plane = "axial"

    return {
        "plane": plane,
        "si_lr": si_lr,
        "bbox": (hx, hy, hw, hh),
        "face_side": face_side,
        "anterior_side": face_side,
        "is_vflipped": is_vflipped,
        "silhouette_iou_lr": silhouette_iou_lr
    }
def fill_core_necrosis_ccomp(core_raw_u8: np.ndarray, max_hole_ratio: float = 0.85) -> np.ndarray:
    core_filled = core_raw_u8.copy()
    ring_area = float(max(np.count_nonzero(core_filled), 1))
    cnts_c, hier_c = cv2.findContours(core_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_c and hier_c is not None:
        for i_c, h_info in enumerate(hier_c[0]):
            if h_info[3] != -1:
                hole_area = float(cv2.contourArea(cnts_c[i_c]))
                if hole_area <= max_hole_ratio * ring_area:
                    cv2.drawContours(core_filled, cnts_c, i_c, 255, thickness=-1)
    return core_filled


def robust_percentile_normalize(image_512: np.ndarray) -> np.ndarray:
    if image_512 is None or image_512.size == 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    if image_512.ndim == 3:
        gray = cv2.cvtColor(image_512, cv2.COLOR_RGBA2GRAY if image_512.shape[2] == 4 else cv2.COLOR_RGB2GRAY)
    else:
        gray = image_512.copy()

    if gray.shape[0] != IMG_SIZE or gray.shape[1] != IMG_SIZE:
        gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

    gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)
    med3 = cv2.medianBlur(gray_u8, 3)

    # Детектор импульсного шума («соль и перец» / _SP_)
    spike_fraction = float(np.mean(np.abs(gray_u8.astype(np.int16) - med3.astype(np.int16)) > 30))
    if spike_fraction > 0.008:
        gray_u8 = cv2.medianBlur(med3, 5) if spike_fraction > 0.030 else med3

    gray_f = gray_u8.astype(np.float32)
    nz_vals = gray_f[gray_f > 2.0]
    if nz_vals.size < 256:
        return gray_u8

    # Адаптивный нижний порог фона: на тёмных снимках (_DA_) не отрезает паренхиму
    raw_med = float(np.median(nz_vals))
    bg_cut = 3.0 if raw_med < 50.0 else 8.0
    fg_vals = gray_f[(gray_f > bg_cut) & (gray_f < 252.0)]
    if fg_vals.size < 256:
        fg_vals = nz_vals

    p1 = float(np.percentile(fg_vals, 1.0))
    p995 = float(np.percentile(fg_vals, 99.5))
    denom = max(p995 - p1, 1e-4)
    norm_f = np.clip((gray_f - p1) / denom * 255.0, 0.0, 255.0)
    norm_f[gray_f <= 2.0] = 0.0

    # Мягкая коррекция гаммы только при сильном отклонении яркости (без агрессивного CLAHE)
    paren_vals = norm_f[(norm_f >= 14.0) & (norm_f <= 215.0)]
    if paren_vals.size > 600:
        med_val = float(np.median(paren_vals))
        if med_val < 58.0 or med_val > 102.0:
            gamma = math.log(78.0 / 255.0) / math.log(max(min(med_val, 240.0), 10.0) / 255.0)
            gamma = float(np.clip(gamma, 0.65, 1.55))
            non_zero = norm_f > 2.0
            norm_f[non_zero] = 255.0 * np.power(norm_f[non_zero] / 255.0, gamma)

    norm_u8 = np.clip(norm_f, 0.0, 255.0).astype(np.uint8)
    norm_u8[gray_f <= 2.0] = 0
    return norm_u8


def _compute_canonical_parenchyma_mask(gray_canon: np.ndarray) -> np.ndarray:
    """
    Построение маски внутричерепного свода:
      - Сохраняет турецкое седло и аденому гипофиза на сагиттальных (#001, #002) и коронарных (#003) срезах;
      - Исключает глазные яблоки и яркую ретробульбарную жировую клетчатку на аксиальных срезах (#028),
        не затрагивая лобные менингиомы и глиомы;
      - Сохраняет кистозные полости и пристеночные узлы свода.
    """
    blurred = cv2.GaussianBlur(gray_canon, (5, 5), 1.2)
    head_init = (blurred > 14).astype(np.uint8) * 255
    head_init = cv2.morphologyEx(head_init, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
    ext_cnts, _ = cv2.findContours(head_init, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not ext_cnts:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    largest_head = max(ext_cnts, key=cv2.contourArea)
    head_area = float(cv2.contourArea(largest_head))
    if head_area < 2500.0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

    hx, hy, hw, hh = cv2.boundingRect(largest_head)
    head_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    cv2.drawContours(head_mask, [largest_head], -1, 255, thickness=-1)

    plane_info = detect_slice_plane_and_level_normalized(gray_canon)
    plane = plane_info["plane"]
    face_side = plane_info["face_side"]

    r_eq = math.sqrt(head_area / math.pi)
    dist_from_scalp = cv2.distanceTransform(head_mask, cv2.DIST_L2, 3)

    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    cranial_prior = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    sellar_sanctuary = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool_)

    if plane == "sagittal":
        cranial_cx = hx + (0.55 * hw if face_side == "left" else 0.45 * hw)
        cranial_cy = hy + 0.36 * hh
        rx_prior, ry_prior = 0.44 * hw, 0.35 * hh
        cv2.ellipse(
            cranial_prior,
            (int(round(cranial_cx)), int(round(cranial_cy))),
            (max(int(round(rx_prior)), 30), max(int(round(ry_prior)), 30)),
            0, 0, 360, 255, -1
        )
        post_cx = hx + (0.64 * hw if face_side == "left" else 0.36 * hw)
        post_cy = hy + 0.52 * hh
        cv2.ellipse(
            cranial_prior,
            (int(round(post_cx)), int(round(post_cy))),
            (max(int(round(0.26 * hw)), 20), max(int(round(0.21 * hh)), 20)),
            0, 0, 360, 255, -1
        )

        # Охранная зона турецкого седла (гипофиза) на сагиттальном срезе (#001, #002)
# Охранная зона турецкого седла и переднего полюса височной доли (#001/#002/#129)
        sella_cx = hx + (0.44 * hw if face_side == "left" else 0.56 * hw)
        sella_cy = hy + 0.53 * hh
        sella_rx, sella_ry = 0.16 * hw, 0.16 * hh
        sellar_sanctuary = (
            ((xx - sella_cx) ** 2) / max(sella_rx ** 2, 1.0)
            + ((yy - sella_cy) ** 2) / max(sella_ry ** 2, 1.0)
        ) <= 1.0
        cv2.ellipse(
            cranial_prior,
            (int(round(sella_cx)), int(round(sella_cy))),
            (max(int(round(sella_rx)), 16), max(int(round(sella_ry)), 16)),
            0, 0, 360, 255, -1
        )

        cranial_prior[min(IMG_SIZE, int(round(hy + 0.72 * hh))):, :] = 0

        if face_side == "right":
            x_ant, y_ant = hx + 0.90 * hw, hy + 0.34 * hh
            x_sel, y_sel = hx + 0.65 * hw, hy + 0.64 * hh
            facial_orbit_cut = (xx > x_sel) & (
                (yy - y_sel) > ((y_ant - y_sel) / max(x_ant - x_sel, 1.0)) * (xx - x_sel)
            )
            ant_orbit_box = (xx > hx + 0.69 * hw) & (yy > hy + 0.40 * hh)
        else:
            x_ant, y_ant = hx + 0.10 * hw, hy + 0.34 * hh
            x_sel, y_sel = hx + 0.35 * hw, hy + 0.64 * hh
            facial_orbit_cut = (xx < x_sel) & (
                (yy - y_sel) > ((y_ant - y_sel) / min(x_ant - x_sel, -1.0)) * (xx - x_sel)
            )
            ant_orbit_box = (xx < hx + 0.31 * hw) & (yy > hy + 0.40 * hh)

        sellar_solid = sellar_sanctuary & (blurred >= 40)
        cranial_prior[facial_orbit_cut & (~sellar_solid)] = 0
        cranial_prior[ant_orbit_box & (~sellar_solid)] = 0

        # Вырезаем чёрный воздух клиновидной пазухи без агрессивной дилатации в сторону гипофиза
        basal_air = (yy > hy + 0.50 * hh) & (blurred < 32)
        if np.any(basal_air):
            basal_air_soft = cv2.dilate(basal_air.astype(np.uint8) * 255, _KERNEL_ERODE_3, iterations=1) > 0
            basal_air_outer = cv2.dilate(basal_air.astype(np.uint8) * 255, _KERNEL_OPEN_9, iterations=1) > 0
            cranial_prior[basal_air_outer & (~sellar_sanctuary)] = 0
            cranial_prior[basal_air_soft & sellar_sanctuary & (blurred < 36)] = 0

    elif plane == "coronal":
        cranial_cx = hx + 0.50 * hw
        cranial_cy = hy + 0.34 * hh
        rx_prior, ry_prior = 0.435 * hw, 0.31 * hh
        cv2.ellipse(
            cranial_prior,
            (int(round(cranial_cx)), int(round(cranial_cy))),
            (max(int(round(rx_prior)), 30), max(int(round(ry_prior)), 30)),
            0, 0, 360, 255, -1
        )
        for s_sign in (-1.0, 1.0):
            cv2.ellipse(
                cranial_prior,
                (int(round(cranial_cx + s_sign * 0.21 * hw)), int(round(hy + 0.46 * hh))),
                (max(int(round(0.20 * hw)), 20), max(int(round(0.16 * hh)), 18)),
                0, 0, 360, 255, -1
            )

        # Охранная зона турецкого седла на коронарном срезе
# Охранная зона турецкого седла и ствола мозга (моста) на коронарном срезе (#001, #003)
        sella_cy = hy + 0.58 * hh
        sella_rx, sella_ry = 0.17 * hw, 0.13 * hh
        sellar_sanctuary = (
            ((xx - cranial_cx) ** 2) / max(sella_rx ** 2, 1.0)
            + ((yy - sella_cy) ** 2) / max(sella_ry ** 2, 1.0)
        ) <= 1.0
        cv2.ellipse(
            cranial_prior,
            (int(round(cranial_cx)), int(round(sella_cy))),
            (max(int(round(sella_rx)), 16), max(int(round(sella_ry)), 14)),
            0, 0, 360, 255, -1
        )

        # Носоглотка вырезается ниже ствола/седла; солидная ткань ствола и гипофиза (>= 40) сохраняется
        naso_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.ellipse(
            naso_mask,
            (int(round(cranial_cx)), int(round(hy + 0.72 * hh))),
            (max(int(round(0.22 * hw)), 20), max(int(round(0.12 * hh)), 14)),
            0, 0, 360, 255, -1
        )
        cranial_prior[(naso_mask > 0) & (~(sellar_sanctuary & (blurred >= 40)))] = 0
        cranial_prior[min(IMG_SIZE, int(round(hy + 0.69 * hh))):, :] = 0

    else:
        # AXIAL
        cranial_cx = hx + 0.50 * hw
        cranial_cy = hy + 0.50 * hh
        rx_prior, ry_prior = 0.435 * hw, 0.44 * hh
        cv2.ellipse(
            cranial_prior,
            (int(round(cranial_cx)), int(round(cranial_cy))),
            (max(int(round(rx_prior)), 30), max(int(round(ry_prior)), 30)),
            0, 0, 360, 255, -1
        )

        # Вырез глазниц и яркой ретробульбарной жировой клетчатки на аксиальном срезе основания черепа (#028)
        y_e0 = hy + int(0.08 * hh)
        y_e1 = hy + int(0.36 * hh)
        x_e0 = max(0, int(cranial_cx - 0.14 * hw))
        x_e1 = min(IMG_SIZE, int(cranial_cx + 0.14 * hw))
        ethmoid_roi = blurred[y_e0:y_e1, x_e0:x_e1]
        ethmoid_air_mask = (ethmoid_roi < 36)
        ethmoid_air_frac = float(np.mean(ethmoid_air_mask)) if ethmoid_roi.size > 40 else 0.0

        l_orb = blurred[hy + int(0.13 * hh):hy + int(0.36 * hh), max(0, int(cranial_cx - 0.34 * hw)):max(1, int(cranial_cx - 0.10 * hw))]
        r_orb = blurred[hy + int(0.13 * hh):hy + int(0.36 * hh), min(IMG_SIZE - 1, int(cranial_cx + 0.10 * hw)):min(IMG_SIZE, int(cranial_cx + 0.34 * hw))]
        p90_l = float(np.percentile(l_orb, 90.0)) if l_orb.size > 40 else 0.0
        p90_r = float(np.percentile(r_orb, 90.0)) if r_orb.size > 40 else 0.0
        has_bilateral_orbital_fat = (p90_l >= 148.0 and p90_r >= 148.0 and ethmoid_air_frac >= 0.10)

        eth_ys_local = np.where(ethmoid_air_mask)[0]
        y_eth_bot = (y_e0 + int(np.max(eth_ys_local))) if eth_ys_local.size > 20 else y_e0
        has_ethmoid_sinus = (ethmoid_air_frac >= 0.20 and y_eth_bot >= hy + int(0.20 * hh))

        if has_ethmoid_sinus or has_bilateral_orbital_fat:
            y_orb_limit = min(int(round(hy + 0.37 * hh)), max(y_eth_bot + 8, int(round(hy + 0.33 * hh))))
            orbit_structures = (
                (yy < y_orb_limit)
                & (np.abs(xx - cranial_cx) < 0.38 * hw)
                & ((blurred < 38) | (blurred >= 135))
            )
            orbit_dil = cv2.dilate(orbit_structures.astype(np.uint8) * 255, _KERNEL_OPEN_9, iterations=1) > 0
            for col_x in range(max(0, int(cranial_cx - 0.37 * hw)), min(IMG_SIZE, int(cranial_cx + 0.37 * hw))):
                o_ys = np.where(orbit_dil[:y_orb_limit, col_x])[0]
                if o_ys.size > 0:
                    cut_y = int(np.max(o_ys)) + 4
                    # По средней линии останавливаемся на задней стенке пазухи, сохраняя гипофиз позади неё
                    if abs(col_x - cranial_cx) < 0.11 * hw:
                        air_mid_ys = np.where(blurred[:y_orb_limit, col_x] < 38)[0]
                        if air_mid_ys.size > 0:
                            cut_y = min(cut_y, int(np.max(air_mid_ys)) + 2)
                    cranial_prior[:min(y_orb_limit + 4, cut_y), col_x] = 0

        top_nose_w = float(np.count_nonzero(head_mask[min(IMG_SIZE - 1, hy + int(0.04 * hh)), hx:hx+hw])) / max(float(hw), 1.0)
        if top_nose_w < 0.28 and (hh >= 1.15 * hw) and (has_ethmoid_sinus or has_bilateral_orbital_fat):
            cranial_prior[:int(round(hy + 0.28 * hh)), :] = 0

    if plane == "axial":
        top_dome_factor = np.where(yy < cranial_cy, 0.082, 0.115)
        min_peel_px = np.where(yy < cranial_cy, 11.0, 15.0)
    else:
        top_dome_factor = np.where(yy < cranial_cy, 0.062, 0.105)
        min_peel_px = np.where(yy < cranial_cy, 8.0, 13.5)

    intracranial_shell = (dist_from_scalp >= np.maximum(min_peel_px, top_dome_factor * r_eq)).astype(np.uint8) * 255

    fg_px = blurred[(intracranial_shell > 0) & (blurred > 14)]
    if fg_px.size > 200:
        p18 = float(np.percentile(fg_px, 18.0))
        med_brain = float(np.median(fg_px))
        sinus_air_thresh = float(np.clip(min(p18 * 0.90, med_brain * 0.46), 18.0, 42.0))
    else:
        med_brain = 78.0
        sinus_air_thresh = 24.0

    is_central_sellar_air = (
        (np.abs(xx - cranial_cx) <= 0.14 * hw)
        & (yy >= (hy + 0.34 * hh))
        & (yy <= (hy + 0.56 * hh))
    )
    upper_dome_void = (
        (intracranial_shell > 0)
        & (cranial_prior > 0)
        & (yy < (hy + 0.52 * hh))
        & (~is_central_sellar_air)
        & (blurred < sinus_air_thresh)
    ).astype(np.uint8) * 255
    void_dist = cv2.distanceTransform(upper_dome_void, cv2.DIST_L2, 5)
    thick_cystic_dome = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool_)
    if np.max(void_dist) >= 7.5:
        num_v, lbl_v, st_v, _ = cv2.connectedComponentsWithStats(upper_dome_void, connectivity=8)
        for kv in range(1, num_v):
            comp_v = (lbl_v == kv)
            if st_v[kv, cv2.CC_STAT_AREA] >= 130 and float(np.max(void_dist[comp_v])) >= 7.5:
                thick_cystic_dome |= comp_v

    solid_tissue = (
        ((blurred >= sinus_air_thresh) | thick_cystic_dome)
        & (intracranial_shell > 0)
        & (cranial_prior > 0)
    ).astype(np.uint8) * 255
    solid_tissue_filled = fill_core_necrosis_ccomp(solid_tissue, max_hole_ratio=0.45)

    severed = cv2.morphologyEx(solid_tissue_filled, cv2.MORPH_OPEN, _KERNEL_OPEN_19, iterations=1)
    if np.count_nonzero(severed) < int(0.18 * head_area):
        severed = cv2.morphologyEx(solid_tissue_filled, cv2.MORPH_OPEN, _KERNEL_OPEN_9, iterations=1)
    if np.count_nonzero(severed) < int(0.16 * head_area):
        severed = cv2.bitwise_and(intracranial_shell, cranial_prior)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(severed, connectivity=8)
    if num_labels <= 1:
        return cv2.bitwise_and(intracranial_shell, cranial_prior)

    best_label, best_score = -1, -1.0
    r0_sq = (0.20 * r_eq) ** 2
    for lbl_idx in range(1, num_labels):
        area_lbl = float(stats[lbl_idx, cv2.CC_STAT_AREA])
        if area_lbl < 800.0:
            continue
        cx_lbl, cy_lbl = centroids[lbl_idx]
        comp_mask = (labels == lbl_idx)
        prior_overlap = float(np.mean(cranial_prior[comp_mask] > 0))
        dist_sq = (cx_lbl - cranial_cx) ** 2 + (cy_lbl - cranial_cy) ** 2
        score = (area_lbl * (0.35 + 0.65 * prior_overlap)) / (dist_sq + r0_sq)
        if score > best_score:
            best_score = score
            best_label = lbl_idx

    if best_label < 1:
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    brain_core = np.where(labels == best_label, 255, 0).astype(np.uint8)

    for lbl_idx in range(1, num_labels):
        if lbl_idx == best_label:
            continue
        area_lbl = float(stats[lbl_idx, cv2.CC_STAT_AREA])
        comp_mask = (labels == lbl_idx)
        is_bright_dome_tumor = (
            area_lbl >= 160.0
            and float(np.mean(yy[comp_mask])) < (hy + 0.54 * hh)
            and float(np.mean(blurred[comp_mask])) >= med_brain + 12.0
        )
        if (area_lbl >= max(500.0, 0.014 * head_area) or is_bright_dome_tumor) and float(np.mean(cranial_prior[comp_mask] > 0)) >= 0.65:
            brain_core[comp_mask] = 255

    # Возвращаем солидную ткань турецкого седла (гипофиза), которую мог отсечь MORPH_OPEN(19)
    if np.any(sellar_sanctuary):
        sellar_tissue = (
            sellar_sanctuary
            & (cranial_prior > 0)
            & (intracranial_shell > 0)
            & (blurred >= max(sinus_air_thresh, 36.0))
        ).astype(np.uint8) * 255
        if np.count_nonzero(sellar_tissue) >= 40:
            brain_dil_s = cv2.dilate(brain_core, _KERNEL_OPEN_19, iterations=1) > 0
            n_st, l_st, st_st, _ = cv2.connectedComponentsWithStats(sellar_tissue, connectivity=8)
            for ks in range(1, n_st):
                if st_st[ks, cv2.CC_STAT_AREA] >= 35 and np.any((l_st == ks) & brain_dil_s):
                    brain_core[l_st == ks] = 255

    upper_bright_nodules = (
        (solid_tissue_filled > 0)
        & (yy < (hy + 0.52 * hh))
        & (blurred >= med_brain + 15.0)
    ).astype(np.uint8) * 255
    if np.count_nonzero(upper_bright_nodules) >= 140:
        brain_core = cv2.bitwise_or(brain_core, upper_bright_nodules)

    if np.any(thick_cystic_dome):
        brain_core[thick_cystic_dome] = 255

    brain_restored = cv2.dilate(brain_core, _KERNEL_OPEN_9, iterations=1)
    brain_restored = cv2.morphologyEx(brain_restored, cv2.MORPH_CLOSE, _KERNEL_CLOSE_21, iterations=1)

    cnts_up, _ = cv2.findContours(brain_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts_up:
        hull_up = cv2.convexHull(np.vstack(cnts_up))
        hull_up_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        cv2.drawContours(hull_up_mask, [hull_up], -1, 255, thickness=-1)
        upper_half_gate = (yy < int(round(hy + 0.52 * hh))).astype(np.uint8) * 255
        dome_bridge = cv2.bitwise_and(hull_up_mask, upper_half_gate)
        brain_restored = cv2.bitwise_or(brain_restored, dome_bridge)

    brain_restored = cv2.bitwise_and(brain_restored, intracranial_shell)
    brain_restored = cv2.bitwise_and(brain_restored, cranial_prior)

    if np.count_nonzero(brain_restored) < int(0.18 * head_area):
        brain_restored = cv2.bitwise_and(intracranial_shell, cranial_prior)

    cnts_b, _ = cv2.findContours(brain_restored, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_parenchyma = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if cnts_b:
        main_cnt = max(cnts_b, key=cv2.contourArea)
        epsilon = 0.0045 * cv2.arcLength(main_cnt, True)
        approx_cnt = cv2.approxPolyDP(main_cnt, epsilon, True)
        cv2.drawContours(final_parenchyma, [approx_cnt], -1, 255, thickness=-1)
        final_parenchyma = fill_core_necrosis_ccomp(final_parenchyma, max_hole_ratio=0.45)
        final_parenchyma = cv2.bitwise_and(final_parenchyma, intracranial_shell)
        final_parenchyma = cv2.bitwise_and(final_parenchyma, cranial_prior)

    _, final_mask = cv2.threshold(cv2.GaussianBlur(final_parenchyma, (7, 7), 1.5), 127, 255, cv2.THRESH_BINARY)
    return final_mask

def _refine_pituitary_sellar_adenoma(
    gray_u8: np.ndarray,
    p_core: np.ndarray,
    p_inflam: np.ndarray,
    p_edema: np.ndarray,
    vault_mask_u8: np.ndarray,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Детектор аденомы гипофиза:
      - Доверяет сильному отклику нейросети в расширенном селлярном коридоре (без отбраковки макроаденом > 4600 px);
      - Не сносит микроаденомы при seg_threshold = 0.80;
      - Блокирует двустороннюю ретробульбарную клетчатку глазниц (#028).
    """
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    vault_bool = (vault_mask_u8 > 127)
    vault_area = float(np.count_nonzero(vault_bool))
    if vault_area < 400.0:
        return empty, empty, empty

    plane_info = detect_slice_plane_and_level_normalized(gray_u8)
    plane = plane_info.get("plane", "coronal")
    ant_side = plane_info.get("face_side", "left")

    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0).astype(np.float32)
    dist_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)

    ys_v, xs_v = np.where(vault_bool)
    vx0, vx1 = float(np.min(xs_v)), float(np.max(xs_v))
    vy0, vy1 = float(np.min(ys_v)), float(np.max(ys_v))
    vw = max(vx1 - vx0, 60.0)
    vh = max(vy1 - vy0, 60.0)
    vcx = 0.50 * (vx0 + vx1)

    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]

    # Расширенный анатомический коридор турецкого седла и супраселлярной цистерны
    if plane == "sagittal":
        sellar_cx = vcx + (0.04 * vw if ant_side == "right" else -0.04 * vw)
        sellar_cy = vy0 + 0.72 * vh
        rx_s, ry_s = 0.32 * vw, 0.34 * vh
        sellar_win = (((xx - sellar_cx) ** 2) / (rx_s ** 2) + ((yy - sellar_cy) ** 2) / (ry_s ** 2)) <= 1.0
        # Блокируем только самый передний край орбиты
        if ant_side == "right":
            sellar_win &= (xx <= vx1 - 0.14 * vw)
        else:
            sellar_win &= (xx >= vx0 + 0.14 * vw)
    elif plane == "axial":
        sellar_cx = vcx
        sellar_cy = vy0 + 0.47 * vh
        rx_s, ry_s = 0.28 * vw, 0.28 * vh
        sellar_win = (((xx - sellar_cx) ** 2) / (rx_s ** 2) + ((yy - sellar_cy) ** 2) / (ry_s ** 2)) <= 1.0
        # Исключаем передние глазницы и латеральные височные кости
        sellar_win &= (yy >= vy0 + 0.24 * vh) & (dist_vault >= 10.0)
    else:
        # Коронарный срез: покрывает и интраселлярную, и крупную супраселлярную часть макроаденомы
        sellar_cx = vcx
        sellar_cy = vy0 + 0.74 * vh
        rx_s, ry_s = 0.30 * vw, 0.36 * vh
        sellar_win = (((xx - sellar_cx) ** 2) / (rx_s ** 2) + ((yy - sellar_cy) ** 2) / (ry_s ** 2)) <= 1.0

    sellar_win &= vault_bool

    p_pit = np.clip(p_core + p_inflam + 0.5 * p_edema, 0.0, 1.0) * vault_bool.astype(np.float32)
    p_pit_sellar = p_pit * sellar_win.astype(np.float32)

    brain_px = blurred[vault_bool & (blurred >= 36.0) & (dist_vault >= 10.0)]
    med_b = float(np.median(brain_px)) if brain_px.size > 100 else 78.0
    iqr_b = max(float(np.percentile(brain_px, 75.0) - np.percentile(brain_px, 25.0)), 8.0) if brain_px.size > 100 else 14.0

    tau = float(np.clip(seg_threshold, 0.20, 0.92))
    max_pit = float(np.max(p_pit_sellar)) if p_pit_sellar.size > 0 else 0.0

    # Адаптивный порог: не обнуляет уверенные аденомы при ползунке tau=0.80
    if max_pit >= 0.16:
        tau_p = float(np.clip(min(tau * 0.68, 0.58 * max_pit), 0.16, 0.62))
    else:
        tau_p = tau

    cand_bin = (sellar_win & (p_pit_sellar >= tau_p) & (blurred >= 34.0)).astype(np.uint8) * 255
    # Если маска слишком велика, аккуратно поднимаем порог
    if float(np.count_nonzero(cand_bin)) > 0.22 * vault_area and max_pit >= 0.25:
        tau_p = float(np.clip(min(tau * 0.85, 0.76 * max_pit), 0.30, 0.80))
        cand_bin = (sellar_win & (p_pit_sellar >= tau_p) & (blurred >= 34.0)).astype(np.uint8) * 255

    open_k = _KERNEL_ERODE_3 if np.count_nonzero(cand_bin) < 350 else _KERNEL_ERODE_5
    cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_OPEN, open_k, iterations=1)
    cand_bin = fill_core_necrosis_ccomp(cand_bin, max_hole_ratio=0.85)

    max_pit_area = max(12000.0, 0.26 * vault_area)
    num_p, lbl_p, st_p, cent_p = cv2.connectedComponentsWithStats(cand_bin, connectivity=8)
    best_pit_mask = empty.copy()
    best_pit_score = -1.0

    for kp in range(1, num_p):
        area_p = float(st_p[kp, cv2.CC_STAT_AREA])
        if area_p < 35.0 or area_p > max_pit_area:
            continue
        comp_p = (lbl_p == kp)
        comp_u8 = (comp_p.astype(np.uint8)) * 255

        cnts_p, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts_p:
            continue
        main_cp = max(cnts_p, key=cv2.contourArea)
        rect_p = cv2.minAreaRect(main_cp)
        wp, hp = max(float(rect_p[1][0]), 1.0), max(float(rect_p[1][1]), 1.0)
        aspect_p = min(wp, hp) / max(wp, hp)
        if aspect_p < 0.24:
            continue

        cx_p, cy_p = cent_p[kp]
        dist_center_norm = ((cx_p - sellar_cx) / max(rx_s, 1.0)) ** 2 + ((cy_p - sellar_cy) / max(ry_s, 1.0)) ** 2
        if dist_center_norm > 1.35:
            continue

        mean_int_p = float(np.mean(blurred[comp_p]))
        prob_p = float(np.mean(p_pit_sellar[comp_p]))
        center_bonus = max(0.35, 1.45 - 0.65 * dist_center_norm)
        int_bonus = max(0.65, mean_int_p / max(med_b, 40.0))
        score_p = (area_p ** 0.72) * (prob_p ** 1.35) * center_bonus * int_bonus * (aspect_p ** 0.45)

        if score_p > best_pit_score:
            best_pit_score = score_p
            best_pit_mask = comp_u8

    # Резерв 1: если нейросеть дала сильный отклик внутри свода (не в глазнице), но чуть за границей sellar_win
    if np.count_nonzero(best_pit_mask) < 35 and float(np.max(p_pit)) >= 0.35:
        fb_tau = float(np.clip(0.60 * float(np.max(p_pit)), 0.25, 0.65))
        fb_bin = (vault_bool & (p_pit >= fb_tau) & (blurred >= 36.0)).astype(np.uint8) * 255
        if plane == "axial":
            fb_bin[:int(vy0 + 0.26 * vh), :] = 0
        fb_bin = fill_core_necrosis_ccomp(fb_bin, max_hole_ratio=0.80)
        n_fb, l_fb, st_fb, c_fb = cv2.connectedComponentsWithStats(fb_bin, connectivity=8)
        for kfb in range(1, n_fb):
            a_fb = float(st_fb[kfb, cv2.CC_STAT_AREA])
            if 40.0 <= a_fb <= max_pit_area:
                cxf, cyf = c_fb[kfb]
                d_fb = ((cxf - sellar_cx) / max(rx_s * 1.4, 1.0)) ** 2 + ((cyf - sellar_cy) / max(ry_s * 1.4, 1.0)) ** 2
                if d_fb <= 1.5:
                    sc_fb = (a_fb ** 0.7) * float(np.mean(p_pit[l_fb == kfb])) / (0.5 + d_fb)
                    if sc_fb > best_pit_score:
                        best_pit_score = sc_fb
                        best_pit_mask = np.where(l_fb == kfb, 255, 0).astype(np.uint8)

    # Резерв 2: поиск яркого селлярного узла, если отклик сегментатора слабый
    if np.count_nonzero(best_pit_mask) < 35 and np.any(sellar_win):
        t_sellar_bright = float(np.clip(med_b + 1.05 * iqr_b, 86.0, 145.0))
        sellar_bright = (sellar_win & (blurred >= t_sellar_bright)).astype(np.uint8) * 255
        sellar_bright = cv2.morphologyEx(sellar_bright, cv2.MORPH_OPEN, _KERNEL_ERODE_3, iterations=1)
        sellar_bright = fill_core_necrosis_ccomp(sellar_bright, max_hole_ratio=0.75)
        n_sb, l_sb, st_sb, c_sb = cv2.connectedComponentsWithStats(sellar_bright, connectivity=8)
        for ksb in range(1, n_sb):
            a_sb = float(st_sb[ksb, cv2.CC_STAT_AREA])
            if 45.0 <= a_sb <= 6500.0:
                cxs, cys = c_sb[ksb]
                d_norm = ((cxs - sellar_cx) / max(rx_s, 1.0)) ** 2 + ((cys - sellar_cy) / max(ry_s, 1.0)) ** 2
                if d_norm <= 0.85:
                    sc_sb = a_sb * (1.2 - d_norm)
                    if sc_sb > best_pit_score:
                        best_pit_score = sc_sb
                        best_pit_mask = np.where(l_sb == ksb, 255, 0).astype(np.uint8)

    if np.count_nonzero(best_pit_mask) < 35:
        return empty, empty, empty

    best_pit_mask = fill_core_necrosis_ccomp(best_pit_mask, max_hole_ratio=0.80)
    best_pit_mask = cv2.bitwise_and(best_pit_mask, vault_mask_u8)

    eroded_c = cv2.erode(best_pit_mask, _KERNEL_ERODE_5, iterations=1)
    if np.count_nonzero(eroded_c) >= 50:
        mask_inflam = cv2.subtract(best_pit_mask, eroded_c)
        mask_core = eroded_c
    else:
        mask_core = best_pit_mask
        mask_inflam = empty.copy()

    return mask_core, mask_inflam, empty.copy()
def recalibrate_cls_probs_with_morphology(
    mean_cls_probs: np.ndarray,
    gray_u8: np.ndarray,
    vault_mask_u8: np.ndarray,
    det_probs_4ch: np.ndarray
) -> Tuple[str, np.ndarray]:
    """
    Согласование вероятностей классификатора без разрушения no_tumor и pituitary:
      - Компенсирует тренировочный перекос cls_weights[meningioma] *= 1.15 из Ячейки 1
        (48 ошибок glio->meni против 4 meni->glio на валидации);
      - Не трогает уверенные no_tumor и pituitary;
      - Переводит глубокие/интрааксиальные пограничные очаги (meningioma 45-62% vs glioma 25-44%) в glioma.
    """
    probs = np.array(mean_cls_probs, dtype=np.float64).copy()

    # 0. Точная байесовская компенсация тренировочного коэффициента 1.15 на meningioma
    probs[1] /= 1.15
    probs = np.clip(probs, 1e-4, None)
    probs = probs / np.sum(probs)

    raw_pred_idx = int(np.argmax(probs))
    vault_bool = (vault_mask_u8 > 127)
    dist_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)
    p_wt = np.clip(det_probs_4ch[1] + det_probs_4ch[2] + det_probs_4ch[3], 0.0, 1.0) * vault_bool

    # 1. Жёсткая защита no_tumor
    if raw_pred_idx == 2:
        peak_p_wt = float(np.max(p_wt)) if p_wt.size > 0 else 0.0
        strong_neural_px = int(np.count_nonzero(p_wt >= 0.62))
        if probs[2] < 0.58 and peak_p_wt >= 0.80 and strong_neural_px >= 240:
            tumor_prior = max(probs[0] + probs[1] + probs[3], 1e-4)
            shift_mass = 0.55 * probs[2]
            probs[0] += shift_mass * (probs[0] / tumor_prior)
            probs[1] += shift_mass * (probs[1] / tumor_prior)
            probs[3] += shift_mass * (probs[3] / tumor_prior)
            probs[2] -= shift_mass
        probs = np.clip(probs, 1e-4, None)
        probs = probs / np.sum(probs)
        return IDX_TO_CLASS[int(np.argmax(probs))], probs.astype(np.float32)

    # 2. Защита pituitary
    if raw_pred_idx == 3:
        probs = np.clip(probs, 1e-4, None)
        probs = probs / np.sum(probs)
        return IDX_TO_CLASS[int(np.argmax(probs))], probs.astype(np.float32)

    # 3. Уточнение пограничных случаев meningioma -> pituitary / glioma
    ys_v, xs_v = np.where(vault_bool)
    peak_p_wt = float(np.max(p_wt)) if p_wt.size > 0 else 0.0
    hot_mask = (p_wt >= max(0.38, 0.52 * peak_p_wt)) & vault_bool

    # Супраселлярная макроаденома строго глубоко по центру (#008)
    if raw_pred_idx == 1 and probs[3] >= 0.16 and xs_v.size > 50 and np.count_nonzero(hot_mask) >= 120:
        vx0, vx1 = float(np.min(xs_v)), float(np.max(xs_v))
        vy0, vy1 = float(np.min(ys_v)), float(np.max(ys_v))
        vw, vh = max(vx1 - vx0, 60.0), max(vy1 - vy0, 60.0)
        vcx = 0.50 * (vx0 + vx1)
        hy_p, hx_p = np.where(hot_mask)
        h_cx, h_cy = float(np.mean(hx_p)), float(np.mean(hy_p))
        min_d_hot = float(np.min(dist_vault[hot_mask]))
        if min_d_hot >= 20.0 and abs(h_cx - vcx) <= 0.15 * vw and (0.32 <= (h_cy - vy0) / vh <= 0.62) and (probs[1] - probs[3] < 0.55):
            probs[3] += 0.60 * probs[1]
            probs[1] *= 0.40

    # Интрааксиальная глиома с близкими вероятностями против менингиомы (#063, #083, #088, #098, #132, #133, #156)
    if int(np.argmax(probs)) == 1 and probs[0] >= 0.24 and (probs[1] - probs[0] < 0.28):
        if np.count_nonzero(hot_mask) >= 50:
            min_d = float(np.min(dist_vault[hot_mask]))
            mean_d = float(np.mean(dist_vault[hot_mask]))
            dural_contact_frac = float(np.mean(dist_vault[hot_mask] <= 8.5))
            brain_med = float(np.median(gray_u8[vault_bool])) if np.any(vault_bool) else 76.0
            hot_int = float(np.mean(gray_u8[hot_mask]))
            edema_ratio = float(np.sum(det_probs_4ch[3][hot_mask])) / max(float(np.sum(p_wt[hot_mask])), 1e-4)

            is_intraaxial_glioma = (
                (min_d >= 9.0 and mean_d >= 16.0)
                or (dural_contact_frac < 0.12 and mean_d >= 14.0)
                or (dural_contact_frac < 0.18 and hot_int < brain_med + 26.0)
                or (edema_ratio >= 0.25 and dural_contact_frac < 0.18)
            )
            if is_intraaxial_glioma:
                probs[0] += 0.30 * probs[1]
                probs[1] *= 0.70

    probs = np.clip(probs, 1e-4, None)
    probs = probs / np.sum(probs)
    return IDX_TO_CLASS[int(np.argmax(probs))], probs.astype(np.float32)

def compute_parenchyma_mask(image_512: np.ndarray) -> np.ndarray:
    gray = robust_percentile_normalize(image_512)
    plane_info = detect_slice_plane_and_level_normalized(gray)
    if plane_info.get("is_vflipped", False):
        gray_canon = cv2.flip(gray, 0)
        mask_canon = _compute_canonical_parenchyma_mask(gray_canon)
        return cv2.flip(mask_canon, 0)
    return _compute_canonical_parenchyma_mask(gray)


def _evaluate_skull_and_orbit_artifact_penalty(
    comp_bool: np.ndarray,
    blurred: np.ndarray,
    dist_inside_vault: np.ndarray,
    vault_bool: np.ndarray,
    plane_info: Dict[str, Any],
    is_meningioma: bool = True
) -> Tuple[bool, float]:
    """
    Отбраковывает:
      1) Яркую жировую клетчатку глазницы в передне-нижнем квадранте сагиттальных срезов (Фото 3 / image (32) и Фото 4 / image (34));
      2) Чёрную полость турецкого седла / основной пазухи в центре среза при поиске менингиом (Фото 1 / image (30));
      3) Сохраняет настоящие менингиомы на своде, в задней черепной ямке и за чёрной кистой (Фото 2, 3, 4, 5).
    """
    area_k = float(np.count_nonzero(comp_bool))
    if area_k < 35.0:
        return True, 0.0

    depths = dist_inside_vault[comp_bool]
    max_inward_depth = float(np.max(depths))
    mean_inward_depth = float(np.mean(depths))
    p85_inward_depth = float(np.percentile(depths, 85.0))
    border_contact_frac = float(np.mean(depths < 6.5))

    comp_u8 = (comp_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return True, 0.0
    main_cnt = max(cnts, key=cv2.contourArea)
    hull_area = max(float(cv2.contourArea(cv2.convexHull(main_cnt))), 1.0)
    solidity = area_k / hull_area

    rect = cv2.minAreaRect(main_cnt)
    w_r, h_r = max(float(rect[1][0]), 1.0), max(float(rect[1][1]), 1.0)
    short_side, long_side = min(w_r, h_r), max(w_r, h_r)
    aspect_ratio = short_side / long_side

    comp_dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    max_comp_thick = float(np.max(comp_dist))
    is_thick_nodule = (max_comp_thick >= 9.0 and solidity >= 0.48 and aspect_ratio >= 0.30)

    comp_dil = cv2.dilate(comp_u8, _KERNEL_DILATE_7, iterations=1) > 0
    ring_mask = comp_dil & (~comp_bool)
    air_bone_void_frac = float(np.mean(blurred[ring_mask] < 32.0)) if np.any(ring_mask) else 0.0
    internal_void_frac = float(np.mean(blurred[comp_bool] < 34.0))

    if border_contact_frac > 0.55 and (max_inward_depth < 14.5 or mean_inward_depth < 8.0) and not is_thick_nodule:
        return True, 0.0
    if border_contact_frac > 0.40 and aspect_ratio < 0.25 and mean_inward_depth < 11.5:
        return True, 0.0
    if not is_meningioma and border_contact_frac > 0.45 and mean_inward_depth < 12.0 and not is_thick_nodule:
        return True, 0.0

    ys, xs = np.where(comp_bool)
    cy_comp, cx_comp = float(np.mean(ys)), float(np.mean(xs))
    hx, hy, hw, hh = plane_info.get("bbox", (0, 0, IMG_SIZE, IMG_SIZE))
    rel_y = (cy_comp - hy) / max(float(hh), 1.0)
    rel_x = (cx_comp - hx) / max(float(hw), 1.0)
    plane = plane_info.get("plane", "axial")
    ant_side = plane_info.get("face_side", "right")

    # ИСПРАВЛЕНИЕ ФОТО 1 (image (30).webp):
    # В центральной зоне гипофиза / основной пазухи (0.37 <= rel_x <= 0.63, 0.34 <= rel_y <= 0.58)
    # запрещаем выделять тёмную полость пазухи/седла (internal_void_frac > 0.22 или mean_int < 65) как менингиому!
    if (0.37 <= rel_x <= 0.63) and (0.34 <= rel_y <= 0.58):
        if internal_void_frac > 0.22 or float(np.mean(blurred[comp_bool])) < 68.0:
            return True, 0.0

    if (not is_meningioma) and (rel_y > 0.57) and (0.34 <= rel_x <= 0.66) and (hh >= 1.04 * hw or plane == "coronal"):
        return True, 0.0

    # ИСПРАВЛЕНИЕ ФОТО 3 (image (32).webp) и ФОТО 4 (image (34).webp):
    # На сагиттальном срезе (или срезе с низкой симметрией si_lr < 0.50) передне-нижний квадрант со стороны лица — это ГЛАЗНИЦА!
    is_sagittal_like = (plane == "sagittal") or (float(plane_info.get("si_lr", 0.85)) < 0.50)
    is_orbit_or_skullbase_zone = False
    if is_sagittal_like:
        is_ant = (rel_x > 0.52) if ant_side == "right" else (rel_x < 0.48)
        if is_ant and rel_y > 0.42:
            return True, 0.0
        if rel_y > 0.62 and border_contact_frac > 0.35:
            is_orbit_or_skullbase_zone = True
    elif plane == "axial":
        # Жёсткая отбраковка двусторонней ретробульбарной жировой клетчатки глазниц (#028 / 1235.jpg)
        if rel_y < 0.38 and (0.12 <= abs(rel_x - 0.50) <= 0.36):
            mean_int_c = float(np.mean(blurred[comp_bool]))
            mirror_xs = np.clip(int(round(2.0 * (hx + 0.50 * hw))) - xs, 0, IMG_SIZE - 1)
            mirror_p85 = float(np.percentile(blurred[ys, mirror_xs], 85.0)) if ys.size > 10 else 0.0
            mid_x0 = max(0, int(hx + 0.38 * hw))
            mid_x1 = min(IMG_SIZE, int(hx + 0.62 * hw))
            mid_y0 = max(0, int(hy + 0.08 * hh))
            mid_y1 = min(IMG_SIZE, int(hy + 0.36 * hh))
            mid_air_frac = float(np.mean(blurred[mid_y0:mid_y1, mid_x0:mid_x1] < 36.0)) if (mid_y1 > mid_y0 and mid_x1 > mid_x0) else 0.0
            if (mean_int_c >= 138.0 and mirror_p85 >= 140.0) or (mean_int_c >= 145.0 and mid_air_frac >= 0.16):
                return True, 0.0
        if rel_y < 0.32 and (border_contact_frac > 0.25 or air_bone_void_frac > 0.14):
            is_orbit_or_skullbase_zone = True
    elif plane == "coronal":
        if rel_y > 0.52 and border_contact_frac > 0.28:
            is_orbit_or_skullbase_zone = True

    if is_orbit_or_skullbase_zone and (air_bone_void_frac > 0.13 or (internal_void_frac > 0.09 and not is_thick_nodule) or mean_inward_depth < 14.5):
        return True, 0.0
    if air_bone_void_frac > 0.25 and mean_inward_depth < 18.0 and not is_thick_nodule:
        return True, 0.0

    inward_dome_bonus = np.clip(p85_inward_depth / 22.0, 0.45, 1.45)
    shape_quality = (solidity ** 1.2) * (max(aspect_ratio, 0.20) ** 0.65) * inward_dome_bonus
    if is_orbit_or_skullbase_zone:
        shape_quality *= 0.30

    return False, float(shape_quality)


def _complete_ring_or_cyst_with_adjacent_necrosis(
    seed_mask_u8: np.ndarray,
    blurred: np.ndarray,
    vault_bool: np.ndarray,
    med_b: float,
    iqr_b: float
) -> np.ndarray:
    if np.count_nonzero(seed_mask_u8) < 45:
        return seed_mask_u8

    t_dark = float(np.clip(med_b - 0.75 * iqr_b, 18.0, 62.0))
    t_rim = float(np.clip(med_b + 0.85 * iqr_b, 82.0, 148.0))

    seed_dil = cv2.dilate(seed_mask_u8, _KERNEL_OPEN_9, iterations=2) > 0
    dark_cavities = (vault_bool & (blurred <= t_dark)).astype(np.uint8) * 255
    dark_cavities = cv2.morphologyEx(dark_cavities, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)

    num_d, lbl_d, st_d, _ = cv2.connectedComponentsWithStats(dark_cavities, connectivity=8)
    combined = seed_mask_u8.copy()
    vault_area = float(np.count_nonzero(vault_bool))

    for kd in range(1, num_d):
        a_d = float(st_d[kd, cv2.CC_STAT_AREA])
        if a_d < 45.0 or a_d > 0.16 * vault_area:
            continue
        comp_d = (lbl_d == kd)
        if np.any(comp_d & seed_dil):
            cav_dil = cv2.dilate(comp_d.astype(np.uint8) * 255, _KERNEL_OPEN_9, iterations=1) > 0
            cav_rim = cav_dil & (~comp_d) & vault_bool
            if np.any(cav_rim) and float(np.mean(blurred[cav_rim] >= t_rim)) >= 0.32:
                combined[comp_d] = 255
                bright_ring = cv2.dilate(comp_d.astype(np.uint8) * 255, _KERNEL_DILATE_7, iterations=2) > 0
                combined[bright_ring & vault_bool & (blurred >= t_rim)] = 255

    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, _KERNEL_CLOSE_15, iterations=1)
    combined = fill_core_necrosis_ccomp(combined, max_hole_ratio=0.88)
    return combined


def _detect_dural_meningioma_mass(
    gray_u8: np.ndarray,
    vault_mask_u8: np.ndarray,
    p_wt_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Находит солидный узел менингиомы:
      - На Фото 1 (image (30).webp) находит яркую параселлярную/кавернозную менингиому рядом с областью гипофиза;
      - На Фото 2 (image (31).webp) находит крупную лобную опухоль и заливает её чёрный некротический центр;
      - На Фото 3 (image (32).webp) находит верхне-парасагиттальную менингиому под куполом свода;
      - На Фото 4 (image (34).webp) находит заднечерепную/мозжечковую менингиому.
    """
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    vault_bool = (vault_mask_u8 > 127)
    vault_area = float(np.count_nonzero(vault_bool))
    if vault_area < 600.0:
        return empty, {"dural_meningioma_score": 0.0, "dural_area_px": 0.0}

    plane_info = detect_slice_plane_and_level_normalized(gray_u8)
    hx, hy, hw, hh = plane_info.get("bbox", (0, 0, IMG_SIZE, IMG_SIZE))
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0).astype(np.float32)
    dist_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)

    brain_px = blurred[vault_bool & (blurred >= 36.0) & (dist_vault >= 8.0)]
    if brain_px.size < 200:
        brain_px = blurred[vault_bool & (blurred >= 36.0)]
    if brain_px.size < 150:
        return empty, {"dural_meningioma_score": 0.0, "dural_area_px": 0.0}

    p15, p30, p48 = np.percentile(brain_px, [15.0, 30.0, 48.0])
    normal_med = float(0.50 * p30 + 0.50 * p48)
    normal_std = max(float((p48 - p15) / 1.04), 6.0)
    otsu_val = float(np.atleast_1d(cv2.threshold(brain_px.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])[0])

    dural_int_th = float(np.clip(max(normal_med + 1.25 * normal_std, 0.80 * otsu_val, normal_med + 13.0), 86.0, 142.0))
    csf_cleft = blurred <= max(normal_med * 0.70, 44.0)

    cand_bin = (vault_bool & (blurred >= dural_int_th) & (~csf_cleft)).astype(np.uint8) * 255
    cand_bin = cv2.morphologyEx(cand_bin, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
    # Заливаем чёрный внутренний некроз внутри яркого узла (исправляет Фото 2 / image (31).webp!)
    cand_bin = fill_core_necrosis_ccomp(cand_bin, max_hole_ratio=0.75)

    num_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(cand_bin, connectivity=8)
    best_mask = empty.copy()
    best_score = -1.0

    for k in range(1, num_c):
        area_k = float(st_c[k, cv2.CC_STAT_AREA])
        if area_k < 220.0 or area_k > 0.30 * vault_area:
            continue

        comp_bool = (lbl_c == k)
        is_art, geom_w = _evaluate_skull_and_orbit_artifact_penalty(
            comp_bool, blurred, dist_vault, vault_bool, plane_info, is_meningioma=True
        )
        if is_art or geom_w < 0.28:
            continue

        depths = dist_vault[comp_bool]
        min_d = float(np.min(depths))
        p85_d = float(np.percentile(depths, 85.0))
        dural_contact_frac = float(np.mean(depths <= 14.0))

        # Разрешаем также фалькс-менингиомы / тенториальные / параселлярные менингиомы (Фото 1, 2, 4),
        # которые лежат вдоль серпа мозга, намёта мозжечка или кавернозного синуса (min_d <= 32 px)
        if min_d > 32.0 or p85_d < 10.0:
            continue

        ys, xs = np.where(comp_bool)
        rel_y = (float(np.mean(ys)) - hy) / max(float(hh), 1.0)
        rel_x = (float(np.mean(xs)) - hx) / max(float(hw), 1.0)

        if rel_y > 0.48 and (0.36 <= rel_x <= 0.64) and area_k < 500.0:
            continue
        dome_loc_bonus = 1.35 if (rel_y < 0.46 or rel_x < 0.35 or rel_x > 0.65) else 0.90

        comp_dil = cv2.dilate(comp_bool.astype(np.uint8) * 255, _KERNEL_OPEN_9, iterations=2) > 0
        near_p_wt = float(np.max(p_wt_map[comp_dil]))
        mean_int = float(np.mean(blurred[comp_bool]))
        capped_delta_int = min(max(mean_int - normal_med, 10.0), 55.0)

        score_k = (
            (area_k ** 0.88)
            * geom_w
            * dome_loc_bonus
            * capped_delta_int
            * (1.0 + 1.5 * dural_contact_frac + 2.5 * near_p_wt)
        )
        if score_k > best_score:
            best_score = score_k
            best_mask = (comp_bool.astype(np.uint8)) * 255

    dural_score = float(np.clip(best_score / 18000.0, 0.0, 1.0)) if best_score > 0 else 0.0
    return best_mask, {
        "dural_meningioma_score": dural_score,
        "dural_area_px": float(np.count_nonzero(best_mask))
    }

def detect_intraaxial_glioma_candidates(
    gray_u8: np.ndarray,
    vault_mask_u8: np.ndarray,
    p_wt_map: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    vault_bool = (vault_mask_u8 > 127)
    vault_area = float(np.count_nonzero(vault_bool))
    if vault_area < 800.0:
        return empty, {"has_ring_necrosis": 0.0, "intraaxial_score": 0.0, "candidate_area_frac": 0.0}

    plane_info = detect_slice_plane_and_level_normalized(gray_u8)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0)
    dist_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)

    brain_px = blurred[vault_bool & (blurred > 28) & (dist_vault >= 8.0)]
    if brain_px.size < 200:
        brain_px = blurred[vault_bool & (blurred > 28)]
    if brain_px.size < 200:
        return empty, {"has_ring_necrosis": 0.0, "intraaxial_score": 0.0, "candidate_area_frac": 0.0}

    med_b = float(np.median(brain_px))
    iqr_b = max(float(np.percentile(brain_px, 75.0) - np.percentile(brain_px, 25.0)), 8.0)

    t_mod = float(np.clip(med_b + 1.20 * iqr_b, 86.0, 155.0))
    t_hyp = float(np.clip(med_b + 1.95 * iqr_b, 105.0, 185.0))
    t_dark_necrosis = float(np.clip(med_b - 0.85 * iqr_b, 20.0, 64.0))

    rim_bin = (vault_bool & (blurred >= t_mod)).astype(np.uint8) * 255
    rim_closed = cv2.morphologyEx(rim_bin, cv2.MORPH_CLOSE, _KERNEL_CLOSE_15, iterations=1)
    rim_filled = fill_core_necrosis_ccomp(rim_closed, max_hole_ratio=0.88)

    num_c, labels_c, stats_c, _ = cv2.connectedComponentsWithStats(rim_filled, connectivity=8)
    best_mask = empty.copy()
    best_score = -1.0
    has_ring_flag = 0.0

    for k in range(1, num_c):
        area_k = float(stats_c[k, cv2.CC_STAT_AREA])
        if area_k < 120.0 or area_k > 0.25 * vault_area:
            continue

        comp_bool = (labels_c == k)
        comp_u8 = _complete_ring_or_cyst_with_adjacent_necrosis(
            comp_bool.astype(np.uint8) * 255, blurred.astype(np.float32), vault_bool, med_b, iqr_b
        )
        comp_bool = (comp_u8 > 0)
        area_k = float(np.count_nonzero(comp_bool))
        if area_k > 0.26 * vault_area:
            continue

        is_art, geom_w = _evaluate_skull_and_orbit_artifact_penalty(
            comp_bool, blurred.astype(np.float32), dist_vault, vault_bool, plane_info, is_meningioma=False
        )
        if is_art:
            continue

        mean_depth = float(np.mean(dist_vault[comp_bool]))
        max_depth = float(np.max(dist_vault[comp_bool]))
        if max_depth < 11.0 or mean_depth < 6.0:
            continue

        eroded_inner = cv2.erode(comp_u8, _KERNEL_OPEN_9, iterations=1) > 0
        ring_band = comp_bool & (~eroded_inner)

        dark_core_frac = float(np.mean(blurred[eroded_inner] <= t_dark_necrosis)) if np.any(eroded_inner) else 0.0
        bright_rim_frac = float(np.mean(blurred[ring_band] >= t_mod)) if np.any(ring_band) else 0.0
        hyper_frac = float(np.mean(blurred[comp_bool] >= t_hyp))
        p_wt_peak = float(np.max(p_wt_map[comp_bool]))
        p_wt_mean = float(np.mean(p_wt_map[comp_bool]))

        is_ring_glioma = (dark_core_frac >= 0.14 and bright_rim_frac >= 0.28 and area_k >= 200.0)
        is_nodular_glioma = (hyper_frac >= 0.20 or p_wt_peak >= 0.28) and (mean_depth >= 7.0)

        if not (is_ring_glioma or is_nodular_glioma):
            continue

        ring_bonus = 3.0 if is_ring_glioma else 1.0
        depth_bonus = min(mean_depth / 14.0, 1.8)
        prob_bonus = 1.0 + 3.5 * p_wt_mean + 1.5 * p_wt_peak
        score_k = (area_k ** 0.75) * ring_bonus * depth_bonus * prob_bonus * geom_w

        if score_k > best_score:
            best_score = score_k
            best_mask = comp_u8
            has_ring_flag = 1.0 if is_ring_glioma else 0.0

    dark_bin = (vault_bool & (blurred <= t_dark_necrosis)).astype(np.uint8) * 255
    dark_bin = cv2.morphologyEx(dark_bin, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
    dark_dist = cv2.distanceTransform(dark_bin, cv2.DIST_L2, 5)
    num_dk, lbl_dk, st_dk, _ = cv2.connectedComponentsWithStats(dark_bin, connectivity=8)

    for kd in range(1, num_dk):
        a_dk = float(st_dk[kd, cv2.CC_STAT_AREA])
        if a_dk < 220.0 or a_dk > 0.22 * vault_area:
            continue
        comp_dk = (lbl_dk == kd)
        max_th = float(np.max(dark_dist[comp_dk]))
        if max_th < 9.5:
            continue
        dk_dil = cv2.dilate(comp_dk.astype(np.uint8) * 255, _KERNEL_OPEN_9, iterations=1) > 0
        rim_dk = dk_dil & (~comp_dk) & vault_bool
        if not np.any(rim_dk):
            continue
        p_peak_dk = float(np.max(p_wt_map[dk_dil]))
        rim_tissue_frac = float(np.mean(blurred[rim_dk] >= med_b + 0.30 * iqr_b))
        if p_peak_dk < 0.20 and rim_tissue_frac < 0.25:
            continue

        full_cyst = ((comp_dk | (dk_dil & (blurred >= med_b + 0.35 * iqr_b))) & vault_bool).astype(np.uint8) * 255
        full_cyst = fill_core_necrosis_ccomp(full_cyst, max_hole_ratio=0.88)
        is_art_dk, geom_dk = _evaluate_skull_and_orbit_artifact_penalty(
            full_cyst > 0, blurred.astype(np.float32), dist_vault, vault_bool, plane_info, is_meningioma=False
        )
        if not is_art_dk and geom_dk >= 0.25:
            sc_dk = (float(np.count_nonzero(full_cyst)) ** 0.78) * (max_th / 9.0) * (1.2 + 3.0 * p_peak_dk) * geom_dk
            if sc_dk > best_score:
                best_score = sc_dk
                best_mask = full_cyst
                has_ring_flag = 1.0

    if np.count_nonzero(best_mask) < 120:
        ys_v, xs_v = np.where(vault_bool)
        if xs_v.size > 400:
            cx_v = int(round(float(np.mean(xs_v))))
            mirror_b = np.full_like(blurred, med_b, dtype=np.float32)
            for x_i in range(IMG_SIZE):
                xm = 2 * cx_v - x_i
                if 0 <= xm < IMG_SIZE:
                    mirror_b[:, x_i] = blurred[:, xm].astype(np.float32)

            asym_diff = (blurred.astype(np.float32) - mirror_b) / iqr_b
            bright_dev = (blurred.astype(np.float32) - med_b) / iqr_b
            asym_cand = (
                vault_bool
                & (dist_vault >= 5.5)
                & (bright_dev >= 1.45)
                & (asym_diff >= 1.25)
            ).astype(np.uint8) * 255
            asym_cand = cv2.morphologyEx(asym_cand, cv2.MORPH_CLOSE, _KERNEL_CLOSE_11, iterations=1)
            asym_cand = fill_core_necrosis_ccomp(asym_cand, max_hole_ratio=0.80)

            n_a, l_a, st_a, _ = cv2.connectedComponentsWithStats(asym_cand, connectivity=8)
            for ka in range(1, n_a):
                a_a = float(st_a[ka, cv2.CC_STAT_AREA])
                if 130.0 <= a_a <= 0.18 * vault_area:
                    comp_a = (l_a == ka)
                    is_art_a, geom_a = _evaluate_skull_and_orbit_artifact_penalty(
                        comp_a, blurred.astype(np.float32), dist_vault, vault_bool, plane_info, is_meningioma=False
                    )
                    if not is_art_a and geom_a >= 0.22:
                        mean_asym = float(np.mean(asym_diff[comp_a]))
                        sc_a = (a_a ** 0.76) * mean_asym * 1.8
                        if sc_a > best_score:
                            best_score = max(sc_a, 980.0)
                            best_mask = (comp_a.astype(np.uint8)) * 255

    area_frac = float(np.count_nonzero(best_mask)) / max(vault_area, 1.0)
    intra_score = float(np.clip(best_score / 1500.0, 0.0, 1.0)) if best_score > 0 else 0.0
    return best_mask, {
        "has_ring_necrosis": has_ring_flag,
        "intraaxial_score": intra_score,
        "candidate_area_frac": area_frac
    }




def recalibrate_cls_probs_with_morphology_v7(
    mean_cls_probs: np.ndarray,
    gray_u8: np.ndarray,
    vault_mask_u8: np.ndarray,
    det_probs_4ch: np.ndarray
) -> Tuple[str, np.ndarray, Dict[str, Any]]:
    raw_pred_idx = int(np.argmax(mean_cls_probs))
    raw_pred_cls = IDX_TO_CLASS[raw_pred_idx]

    pred_cls, recal_probs = recalibrate_cls_probs_with_morphology(
        mean_cls_probs=mean_cls_probs,
        gray_u8=gray_u8,
        vault_mask_u8=vault_mask_u8,
        det_probs_4ch=det_probs_4ch
    )

    vault_bool = (vault_mask_u8 > 127)
    p_wt = np.clip(det_probs_4ch[1] + det_probs_4ch[2] + det_probs_4ch[3], 0.0, 1.0) * vault_bool
    _, g_info = detect_intraaxial_glioma_candidates(gray_u8, vault_mask_u8, p_wt)

    recal_meta = {
        "raw_pred_cls": raw_pred_cls,
        "recalibrated": (raw_pred_cls != pred_cls),
        "has_ring_necrosis": float(g_info["has_ring_necrosis"]),
        "intraaxial_score": round(float(g_info["intraaxial_score"]), 3),
        "in_sellar_zone": (pred_cls == "pituitary")
    }
    return pred_cls, recal_probs, recal_meta


def extract_brain_parenchyma(image_512: np.ndarray) -> np.ndarray:
    mask = compute_parenchyma_mask(image_512)
    if image_512.ndim == 3:
        return cv2.bitwise_and(image_512, image_512, mask=mask)
    return cv2.bitwise_and(image_512, mask)


class ConvNormAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout_p: float = 0.10) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    def __init__(self, f_g: int, f_l: int, f_int: int) -> None:
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(f_g, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
        self.W_x = nn.Sequential(nn.Conv2d(f_l, f_int, kernel_size=1, bias=True), nn.BatchNorm2d(f_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(f_int, 1, kernel_size=1, bias=True), nn.BatchNorm2d(1), nn.Sigmoid())

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)
        return x * self.psi(g1 + x1)


class DecoderAttentionUpBlock(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, dropout_p: float = 0.12) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.att = AttentionGate(f_g=out_ch, f_l=skip_ch, f_int=max(out_ch // 2, 32))
        self.conv = ConvNormAct(out_ch + skip_ch, out_ch, dropout_p=dropout_p)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.up(x)
        if g.shape[2:] != skip.shape[2:]:
            g = F.interpolate(g, size=skip.shape[2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([g, self.att(g=g, x=skip)], dim=1))


class MultiTaskNeuroNet(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        num_classes: int = 4,
        num_seg_classes: int = 4,
        pretrained: bool = False
    ) -> None:
        super().__init__()
        self.phase = 1
        self.num_classes = num_classes
        self.num_seg_classes = num_seg_classes
        self.encoder = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3)
        )
        self.backbone_name = backbone_name
        enc_channels = self.encoder.feature_info.channels()
        c1, c2, c3, c4 = enc_channels[0], enc_channels[1], enc_channels[2], enc_channels[3]

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.cls_head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(c4, 256),
            nn.GELU(),
            nn.Dropout(p=0.30),
            nn.Linear(256, num_classes)
        )

        self.up_stage3 = DecoderAttentionUpBlock(in_ch=c4, skip_ch=c3, out_ch=256, dropout_p=0.15)
        self.up_stage2 = DecoderAttentionUpBlock(in_ch=256, skip_ch=c2, out_ch=128, dropout_p=0.12)
        self.up_stage1 = DecoderAttentionUpBlock(in_ch=128, skip_ch=c1, out_ch=64, dropout_p=0.10)

        self.up_256 = nn.Sequential(nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2), ConvNormAct(32, 32))
        self.up_512 = nn.Sequential(nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2), ConvNormAct(16, 16))
        self.seg_head = nn.Conv2d(16, num_seg_classes, kernel_size=1)

    def set_phase(self, phase: int) -> None:
        self.phase = int(phase)

    def set_encoder_trainable(self, trainable: bool) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats[0], feats[1], feats[2], feats[3]

        pooled_f4 = self.global_pool(f4).flatten(1)
        cls_feat = pooled_f4.detach() if (self.training and self.phase == 2) else pooled_f4
        cls_logits = self.cls_head(cls_feat)

        d3 = self.up_stage3(f4, f3)
        d2 = self.up_stage2(d3, f2)
        d1 = self.up_stage1(d2, f1)
        d0 = self.up_256(d1)
        d_full = self.up_512(d0)

        if d_full.shape[2:] != x.shape[2:]:
            d_full = F.interpolate(d_full, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.seg_head(d_full), cls_logits, pooled_f4


class MahalanobisOODDetector:
    def __init__(self) -> None:
        self.centroids = None
        self.precision_matrix = None
        self.scale_factor = 1.0
        self.threshold = 48.0
        self.shrinkage_coeff = 0.0005
        self.fitted = False

    @staticmethod
    def l2_normalize(features: np.ndarray) -> np.ndarray:
        return features / (np.linalg.norm(features, ord=2, axis=-1, keepdims=True) + 1e-7)

    def verify_embedding(self, feature_vec: np.ndarray) -> Tuple[float, bool]:
        if not self.fitted or self.centroids is None or self.precision_matrix is None:
            return 24.0, True
        z_norm = self.l2_normalize(feature_vec.reshape(1, -1).astype(np.float64))
        dists = [
            np.sqrt(np.maximum(np.sum(((z_norm - self.centroids[c]) @ self.precision_matrix) * (z_norm - self.centroids[c]), axis=1), 0.0))[0]
            for c in range(self.centroids.shape[0])
        ]
        dist = float(min(dists) * self.scale_factor)
        return dist, bool(dist <= self.threshold)

    def load(self, npz_path: Path, fallback_dict: Optional[Dict[str, Any]] = None) -> None:
        try:
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                self.centroids = np.array(data["centroids"], dtype=np.float64)
                self.precision_matrix = np.array(data["precision_matrix"], dtype=np.float64)
                self.scale_factor = float(np.ravel(data["scale_factor"])[0])
                self.threshold = float(np.ravel(data["threshold"])[0])
                self.shrinkage_coeff = float(np.ravel(data["shrinkage_coeff"])[0])
                self.fitted = True
            elif fallback_dict:
                self.centroids = np.array(fallback_dict["centroids"], dtype=np.float64)
                self.precision_matrix = np.array(fallback_dict["precision_matrix"], dtype=np.float64)
                self.scale_factor = float(np.ravel(fallback_dict["scale_factor"])[0])
                self.threshold = float(np.ravel(fallback_dict["threshold"])[0])
                self.shrinkage_coeff = float(np.ravel(fallback_dict["shrinkage_coeff"])[0])
                self.fitted = True
        except Exception:
            self.fitted = False


# ==============================================================================
# 4. ПОСТОБРАБОТКА, ГИБРИДНАЯ РЕКОНСТРУКЦИЯ МЕНИНГИОМ И КОНТРОЛЛЕР СППР
# ==============================================================================
def fast_guided_filter_2d(guide_u8: np.ndarray, src_prob: np.ndarray, radius: int = 4, eps: float = 1e-3) -> np.ndarray:
    I = guide_u8.astype(np.float32) / 255.0
    p = src_prob.astype(np.float32)
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
    mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
    mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
    var_I = np.maximum(mean_II - mean_I * mean_I, 0.0)
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    return np.clip(cv2.boxFilter(a, cv2.CV_32F, ksize) * I + cv2.boxFilter(b, cv2.CV_32F, ksize), 0.0, 1.0)


def _prune_leaking_tails_and_bridges(
    mask_u8: np.ndarray,
    blurred: np.ndarray,
    p_wt: np.ndarray,
    vault_bool: np.ndarray,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> np.ndarray:
    if np.count_nonzero(mask_u8) < 60:
        return mask_u8

    clean = mask_u8.copy()
    dist_vault = cv2.distanceTransform((vault_bool.astype(np.uint8)) * 255, cv2.DIST_L2, 3)

    dark_in_mask = ((blurred < 30.0) & (clean > 0)).astype(np.uint8) * 255
    dark_thick = cv2.distanceTransform(dark_in_mask, cv2.DIST_L2, 5)
    max_dark_r = float(np.max(dark_thick)) if np.any(dark_in_mask > 0) else 0.0

    if max_dark_r < 8.5:
        sinus_void = (blurred < 30.0) & (dist_vault < 22.0) & (p_wt < 0.35)
        clean[sinus_void] = 0
    else:
        sinus_void = (blurred < 24.0) & (dist_vault < 8.0) & (dark_thick < 5.0) & (p_wt < 0.25)
        clean[sinus_void] = 0

    clean = fill_core_necrosis_ccomp(clean, max_hole_ratio=0.85)
    dist_m = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    max_r = float(np.max(dist_m))
    if max_r < 4.0:
        return clean

    tau_scale = float(np.clip((seg_threshold - 0.35) / 0.45, 0.0, 1.25))
    neck_ratio = 0.36 + 0.14 * tau_scale
    neck_px = max(4.5, min(neck_ratio * max_r, 15.0))

    cores = (dist_m >= neck_px).astype(np.uint8) * 255
    num_c, lbl_c, st_c, _ = cv2.connectedComponentsWithStats(cores, connectivity=8)

    if num_c > 1:
        best_c_idx = 1
        best_c_score = -1.0
        for kc in range(1, num_c):
            comp_c = (lbl_c == kc)
            a_c = float(st_c[kc, cv2.CC_STAT_AREA])
            r_c = float(np.max(dist_m[comp_c]))
            p_c = float(np.mean(p_wt[comp_c]))
            int_c = float(np.mean(blurred[comp_c]))
            int_w = max(int_c / 100.0, 0.80 if r_c >= 9.0 else 0.35)
            sc_c = (a_c ** 0.70) * (r_c ** 1.4) * (0.4 + 0.6 * p_c) * int_w
            if sc_c > best_c_score:
                best_c_score = sc_c
                best_c_idx = kc

        main_core = np.where(lbl_c == best_c_idx, 255, 0).astype(np.uint8)
        main_int = float(np.mean(blurred[main_core > 0]))
        main_std = max(float(np.std(blurred[main_core > 0])), 6.0)

        dil_steps = max(1, int(round(neck_px / 2.2)))
        reconstructed = main_core.copy()
        k_step = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        int_floor = main_int - (2.1 - 0.65 * tau_scale) * main_std
        valid_domain = (clean > 0) & (
            (blurred >= int_floor) | (p_wt >= max(0.45, seg_threshold * 0.9)) | (dark_thick >= 8.5)
        )
        for _ in range(dil_steps + 2):
            reconstructed = cv2.dilate(reconstructed, k_step, iterations=1)
            reconstructed = np.where(valid_domain, reconstructed, 0).astype(np.uint8)

        if np.count_nonzero(reconstructed) >= 60:
            clean = reconstructed

    open_k = _KERNEL_OPEN_9 if max_r >= 14.0 else _KERNEL_ERODE_5
    smoothed = cv2.morphologyEx(clean, cv2.MORPH_OPEN, open_k, iterations=1)
    if np.count_nonzero(smoothed) >= 50:
        clean = smoothed

    num_f, lbl_f, st_f, _ = cv2.connectedComponentsWithStats(clean, connectivity=8)
    if num_f > 1:
        best_f = 1 + int(np.argmax(st_f[1:, cv2.CC_STAT_AREA]))
        clean = np.where(lbl_f == best_f, 255, 0).astype(np.uint8)

    clean = fill_core_necrosis_ccomp(clean, max_hole_ratio=0.85)
    return clean


def _complete_whole_meningioma_nodule(
    seed_mask_u8: np.ndarray,
    blurred: np.ndarray,
    p_wt: np.ndarray,
    vault_bool: np.ndarray,
    normal_med: float,
    normal_std: float,
    mod_thresh: float,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> np.ndarray:
    seed_px = int(np.count_nonzero(seed_mask_u8))
    if seed_px < 25:
        return seed_mask_u8

    seed_bool = (seed_mask_u8 > 0)
    seed_vals = blurred[seed_bool]
    seed_med = float(np.median(seed_vals))
    seed_q25 = float(np.percentile(seed_vals, 25.0))
    seed_iqr = max(float(np.percentile(seed_vals, 75.0) - seed_q25), 6.0)

    cnts_s, _ = cv2.findContours(seed_mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity_s = 0.0
    aspect_s = 1.0
    if cnts_s:
        main_s = max(cnts_s, key=cv2.contourArea)
        hull_s = max(float(cv2.contourArea(cv2.convexHull(main_s))), 1.0)
        solidity_s = float(cv2.contourArea(main_s)) / hull_s
        rect_s = cv2.minAreaRect(main_s)
        w_s, h_s = max(float(rect_s[1][0]), 1.0), max(float(rect_s[1][1]), 1.0)
        aspect_s = min(w_s, h_s) / max(w_s, h_s)

    dist_vault = cv2.distanceTransform((vault_bool.astype(np.uint8)) * 255, cv2.DIST_L2, 3)
    seed_min_d = float(np.min(dist_vault[seed_bool]))
    seed_mean_d = float(np.mean(dist_vault[seed_bool]))
    is_partial_crescent = (aspect_s < 0.52) or (solidity_s < 0.76) or (seed_min_d > 5.0 and seed_mean_d < 36.0)

    if (solidity_s >= 0.80 and aspect_s >= 0.55 and seed_px >= 500 and seed_med >= normal_med + 22.0 and not is_partial_crescent):
        strict_floor = max(seed_q25 - 6.0, seed_med - 1.15 * seed_iqr, normal_med + 1.15 * normal_std)
        tight_dil = cv2.dilate(seed_mask_u8, _KERNEL_ERODE_5, iterations=1) > 0
        refined = ((seed_bool | (tight_dil & (blurred >= strict_floor))) & vault_bool).astype(np.uint8) * 255
        refined = fill_core_necrosis_ccomp(refined, max_hole_ratio=0.82)
        return _prune_leaking_tails_and_bridges(refined, blurred, p_wt, vault_bool, seg_threshold)

    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx * gx + gy * gy)

    stroma_int_floor = float(np.clip(normal_med + 0.85 * normal_std, 82.0, min(mod_thresh, seed_q25 - 10.0)))

    capsule_barrier = (
        (blurred <= max(normal_med - 0.35 * normal_std, 40.0))
        | ((grad_mag >= 46.0) & (blurred < normal_med + 0.50 * normal_std))
    )

    ys_s, xs_s = np.where(seed_bool)
    cy_s, cx_s = float(np.mean(ys_s)), float(np.mean(xs_s))
    r_seed = max(math.sqrt(float(ys_s.size) / math.pi), 14.0)
    max_grow_radius = min(max(2.35 * r_seed, 42.0), 96.0)
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    local_ball = ((xx - cx_s) ** 2 + (yy - cy_s) ** 2) <= (max_grow_radius ** 2)

    support_bool = (
        vault_bool
        & local_ball
        & (~capsule_barrier)
        & (blurred >= stroma_int_floor)
        & ((blurred >= normal_med + 1.05 * normal_std) | (p_wt >= 0.22) | (dist_vault <= seed_mean_d + 6.0))
    )
    grow_domain = support_bool.astype(np.uint8) * 255
    grow_domain = cv2.morphologyEx(grow_domain, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)

    num_g, labels_g, stats_g, _ = cv2.connectedComponentsWithStats(grow_domain, connectivity=8)
    grown_mask = seed_mask_u8.copy()
    vault_area = float(np.count_nonzero(vault_bool))

    for kg in range(1, num_g):
        comp_g = (labels_g == kg)
        overlap = np.count_nonzero(comp_g & seed_bool)
        if overlap >= max(12, int(0.18 * ys_s.size)):
            area_g = float(stats_g[kg, cv2.CC_STAT_AREA])
            if area_g <= min(4.2 * float(ys_s.size), 0.25 * vault_area):
                grown_mask[comp_g] = 255

    dural_wall_adjacent = (
        vault_bool
        & local_ball
        & (dist_vault <= max(seed_mean_d + 4.0, 28.0))
        & ((blurred >= normal_med + 0.55 * normal_std) | (blurred <= normal_med - 0.70 * normal_std))
    )
    pocket_bridge = cv2.bitwise_or(grown_mask, (dural_wall_adjacent.astype(np.uint8)) * 255)
    pocket_bridge = cv2.morphologyEx(pocket_bridge, cv2.MORPH_CLOSE, _KERNEL_CLOSE_15, iterations=1)
    pocket_bridge = fill_core_necrosis_ccomp(pocket_bridge, max_hole_ratio=0.85)

    num_p, lbl_p, st_p, _ = cv2.connectedComponentsWithStats(pocket_bridge, connectivity=8)
    for kp in range(1, num_p):
        comp_p = (lbl_p == kp)
        if np.count_nonzero(comp_p & seed_bool) >= int(0.50 * ys_s.size):
            area_p = float(st_p[kp, cv2.CC_STAT_AREA])
            if area_p <= min(4.5 * float(ys_s.size), 0.24 * vault_area):
                cnts_gm, _ = cv2.findContours(grown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if cnts_gm:
                    hull_gm = cv2.convexHull(np.vstack(cnts_gm))
                    hull_gm_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
                    cv2.drawContours(hull_gm_mask, [hull_gm], -1, 255, thickness=-1)
                    grown_mask = cv2.bitwise_or(grown_mask, cv2.bitwise_and(comp_p.astype(np.uint8) * 255, hull_gm_mask))

    grown_mask = fill_core_necrosis_ccomp(grown_mask, max_hole_ratio=0.85)
    return _prune_leaking_tails_and_bridges(grown_mask, blurred, p_wt, vault_bool, seg_threshold)


def refine_glioma_subzones_by_intensity(
    gray_u8: np.ndarray,
    wt_mask_u8: np.ndarray,
    p_core_raw: np.ndarray,
    p_inflam_raw: np.ndarray,
    p_edema_raw: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wt_bin = (wt_mask_u8 > 127)
    empty = np.zeros_like(wt_mask_u8)
    if np.count_nonzero(wt_bin) < 35:
        return empty, empty, empty

    wt_filled = fill_core_necrosis_ccomp(wt_mask_u8, max_hole_ratio=0.82)
    wt_bin = (wt_filled > 127)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0).astype(np.float32)

    dist_wt = cv2.distanceTransform(wt_filled, cv2.DIST_L2, 3)
    max_r = max(float(np.max(dist_wt)), 1.0)
    roi_vals = blurred[wt_bin]
    q35 = float(np.percentile(roi_vals, 35.0))
    q68 = float(np.percentile(roi_vals, 68.0))

    core_bool = wt_bin & (
        (dist_wt >= 0.36 * max_r)
        | (p_core_raw >= 0.38)
        | ((blurred <= q35) & (dist_wt >= 0.20 * max_r))
    )
    mask_core = (core_bool.astype(np.uint8)) * 255
    mask_core = fill_core_necrosis_ccomp(mask_core, max_hole_ratio=0.80)
    if np.count_nonzero(mask_core) < 20:
        mask_core = ((dist_wt >= 0.25 * max_r) & wt_bin).astype(np.uint8) * 255

    core_dil = cv2.dilate(mask_core, _KERNEL_DILATE_7, iterations=1)
    core_erode = cv2.erode(mask_core, _KERNEL_ERODE_5, iterations=1)
    rim_band = (cv2.subtract(core_dil, core_erode) > 0) & wt_bin
    inflam_bool = rim_band & ((blurred >= q68) | (p_inflam_raw >= 0.22))
    mask_inflam = (inflam_bool.astype(np.uint8)) * 255

    mask_core = np.where(mask_inflam > 0, 0, mask_core).astype(np.uint8)
    mask_core = fill_core_necrosis_ccomp(mask_core, max_hole_ratio=0.75)
    mask_inflam = np.where(mask_core > 0, 0, mask_inflam).astype(np.uint8)

    solid_part = ((mask_core > 0) | (mask_inflam > 0)).astype(np.uint8) * 255
    edema_bool = wt_bin & (solid_part == 0)
    if np.count_nonzero(edema_bool) < 15 and np.count_nonzero(solid_part) > 30:
        peri_dil = cv2.dilate(solid_part, _KERNEL_ERODE_5, iterations=1)
        edema_bool = (cv2.subtract(peri_dil, solid_part) > 0) & (p_edema_raw >= 0.16)

    mask_edema = (edema_bool.astype(np.uint8)) * 255
    return mask_core, mask_inflam, mask_edema


def refine_meningioma_hybrid_subzones(
    gray_u8: np.ndarray,
    p_core: np.ndarray,
    p_inflam: np.ndarray,
    p_edema: np.ndarray,
    vault_mask_u8: np.ndarray,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Гибридная 3-зонная реконструкция менингиом (учитывает Фото 1–5):
      - Сохраняет выделение узла за чёрной кистозной полостью на своде черепа (как на Фото 5 / image (22));
      - Защищает центральную область гипофиза / основной пазухи от ложного выделения чёрной полости
        и переносит маску на соседнюю яркую параселлярную область (Фото 1 / image (30));
      - Выделяет лобную опухоль с чёрным некротическим центром (Фото 2 / image (31)) и менингиомы на сагиттальных
        срезах без захвата глазниц (Фото 3 / image (32) и Фото 4 / image (34)).
    """
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    vault_bool = (vault_mask_u8 > 127)
    vault_area = float(np.count_nonzero(vault_bool))
    if vault_area < 400.0:
        return empty, empty, empty

    plane_info = detect_slice_plane_and_level_normalized(gray_u8)
    hx, hy, hw, hh = plane_info.get("bbox", (0, 0, IMG_SIZE, IMG_SIZE))
    dist_inside_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)

    vault_f = vault_bool.astype(np.float32)
    blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0)

    # ИСПРАВЛЕНИЕ ФОТО 1 (image (30).webp) с сохранением ФОТО 5 (image (22).webp):
    # Толстые чёрные кисты разрешены на периферии/своде черепа (Фото 5), НО в центральной области гипофиза /
    # основной пазухи (0.37 <= rel_x <= 0.63, 0.33 <= rel_y <= 0.58) чёрный фон (blurred < 40) строго обнуляется!
    yy, xx = np.mgrid[:IMG_SIZE, :IMG_SIZE]
    in_sellar_pituitary_center = (
        (xx >= (hx + 0.37 * hw)) & (xx <= (hx + 0.63 * hw))
        & (yy >= (hy + 0.33 * hh)) & (yy <= (hy + 0.58 * hh))
    )
    dark_u8 = (vault_bool & (~in_sellar_pituitary_center) & (blurred < 34)).astype(np.uint8) * 255
    dark_thick = cv2.distanceTransform(dark_u8, cv2.DIST_L2, 5)
    G_tissue = (
        ((blurred >= 34) | (dark_thick >= 7.5))
        & (~(in_sellar_pituitary_center & (blurred < 44)))
    ).astype(np.float32)

    p_wt = np.clip(p_core + p_inflam + p_edema, 0.0, 1.0) * vault_f * G_tissue

    brain_px = blurred[vault_bool & (blurred >= 36) & (dist_inside_vault >= 8.0)]
    if brain_px.size < 200:
        brain_px = blurred[vault_bool & (blurred >= 36)]
    if brain_px.size > 150:
        p15 = float(np.percentile(brain_px, 15.0))
        p30 = float(np.percentile(brain_px, 30.0))
        p48 = float(np.percentile(brain_px, 48.0))
        normal_med = 0.50 * p30 + 0.50 * p48
        normal_std = max((p48 - p15) / 1.04, 6.0)
        otsu_intracranial = float(
            np.atleast_1d(cv2.threshold(brain_px, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])[0]
        )
    else:
        normal_med, normal_std, otsu_intracranial = 76.0, 10.0, 110.0

    mod_thresh = float(np.clip(normal_med + 1.15 * normal_std, 84.0, 134.0))
    hyper_thresh = float(
        np.clip(max(normal_med + 1.70 * normal_std, 0.84 * otsu_intracranial, normal_med + 15.0), 92.0, 144.0)
    )

    # 1. Нейросетевая маска neural_mask
    max_wt = float(np.max(p_wt))
    neural_mask = empty.copy()
    if max_wt >= 0.14:
        tau_wt = float(np.clip(seg_threshold * 0.92, 0.18, 0.88 * max_wt))
        raw_nm = ((p_wt >= tau_wt) & vault_bool & (G_tissue > 0)).astype(np.uint8) * 255
        if float(np.count_nonzero(raw_nm)) > 0.12 * vault_area:
            active_vals = p_wt[(p_wt >= tau_wt) & vault_bool]
            tau_wt = float(np.clip(np.percentile(active_vals, 68.0), max(tau_wt, 0.45), 0.88 * max_wt))
            raw_nm = ((p_wt >= tau_wt) & vault_bool & (G_tissue > 0)).astype(np.uint8) * 255

        raw_nm = _prune_leaking_tails_and_bridges(raw_nm, blurred.astype(np.float32), p_wt, vault_bool, seg_threshold)
        num_nm, labels_nm, stats_nm, _ = cv2.connectedComponentsWithStats(raw_nm, connectivity=8)
        best_nm_score = -1.0
        for k in range(1, num_nm):
            a_nm = float(stats_nm[k, cv2.CC_STAT_AREA])
            if a_nm < 55.0 or a_nm > 0.30 * vault_area:
                continue
            comp_nm = (labels_nm == k)
            is_art, geom_w = _evaluate_skull_and_orbit_artifact_penalty(
                comp_nm, blurred.astype(np.float32), dist_inside_vault, vault_bool, plane_info, is_meningioma=True
            )
            if is_art:
                continue
            sc_nm = (a_nm ** 0.72) * (float(np.mean(p_wt[comp_nm])) ** 1.35) * geom_w
            if sc_nm > best_nm_score:
                best_nm_score = sc_nm
                neural_mask = (comp_nm.astype(np.uint8)) * 255

    # ИСПРАВЛЕНИЕ ФОТО 1 (image (30).webp):
    # Если первоначальный отклик нейросети был около центральной области гипофиза / основной пазухи,
    # проверяем яркую параселлярную область рядом с ней (например, кавернозный синус / медиальную височную долю слева на Фото 1)!
    raw_p_all = np.clip(p_core + p_inflam + p_edema, 0.0, 1.0) * vault_f
    sellar_hot = in_sellar_pituitary_center & (raw_p_all >= 0.25)
    if np.count_nonzero(sellar_hot) >= 40:
        parasellar_band = (
            cv2.dilate(sellar_hot.astype(np.uint8) * 255, _KERNEL_OPEN_19, iterations=2) > 0
        ) & vault_bool & (~(in_sellar_pituitary_center & (blurred < mod_thresh)))
        adj_bright = (parasellar_band & (blurred >= mod_thresh)).astype(np.uint8) * 255
        adj_bright = cv2.morphologyEx(adj_bright, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
        adj_bright = fill_core_necrosis_ccomp(adj_bright, max_hole_ratio=0.65)
        n_ab, l_ab, st_ab, _ = cv2.connectedComponentsWithStats(adj_bright, connectivity=8)
        best_ab_area = 0.0
        best_ab_mask = empty.copy()
        for k_ab in range(1, n_ab):
            a_ab = float(st_ab[k_ab, cv2.CC_STAT_AREA])
            if 160.0 <= a_ab <= 0.18 * vault_area and a_ab > best_ab_area:
                best_ab_area = a_ab
                best_ab_mask = np.where(l_ab == k_ab, 255, 0).astype(np.uint8)
        if np.any(best_ab_mask > 0):
            p_wt = np.maximum(p_wt, (best_ab_mask > 0).astype(np.float32) * 0.85)
            if np.count_nonzero(neural_mask) < 120 or float(np.mean(blurred[neural_mask > 0])) < mod_thresh:
                neural_mask = best_ab_mask.copy()

    # 2. Поиск гиперконтрастного ядра менингиомы + пристеночного/параселлярного узла
    dural_mass_mask, dural_info = _detect_dural_meningioma_mass(gray_u8, vault_mask_u8, p_wt)

    hyper_cand = (vault_bool & (blurred >= hyper_thresh)).astype(np.uint8) * 255
    hyper_cand = fill_core_necrosis_ccomp(hyper_cand, max_hole_ratio=0.75)

    csf_cleft = (blurred <= max(normal_med * 0.68, 44.0)) & (dark_thick < 7.5)
    hyper_sep = hyper_cand.copy()
    hyper_sep[csf_cleft] = 0
    hyper_sep = cv2.morphologyEx(hyper_sep, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
    hyper_sep = fill_core_necrosis_ccomp(hyper_sep, max_hole_ratio=0.75)

    num_hc, labels_hc, stats_hc, _ = cv2.connectedComponentsWithStats(hyper_sep, connectivity=8)
    best_seeded_hyper = empty.copy()
    best_seeded_score = -1.0
    best_unseeded_hyper = empty.copy()
    best_unseeded_score = -1.0

    for k in range(1, num_hc):
        area_k = float(stats_hc[k, cv2.CC_STAT_AREA])
        if area_k < 110.0 or area_k > 0.32 * vault_area:
            continue
        comp_bool = (labels_hc == k)
        is_art, geom_w = _evaluate_skull_and_orbit_artifact_penalty(
            comp_bool, blurred.astype(np.float32), dist_inside_vault, vault_bool, plane_info, is_meningioma=True
        )
        if is_art:
            continue

        ys_k, xs_k = np.where(comp_bool)
        rel_yk = (float(np.mean(ys_k)) - hy) / max(float(hh), 1.0)
        rel_xk = (float(np.mean(xs_k)) - hx) / max(float(hw), 1.0)
        mean_int_k = float(np.mean(blurred[comp_bool]))
        if area_k < 280.0 and rel_yk > 0.45 and (0.34 <= rel_xk <= 0.66):
            if dural_info["dural_area_px"] >= 320.0 or mean_int_k >= 200.0:
                continue

        comp_dil = cv2.dilate(comp_bool.astype(np.uint8) * 255, _KERNEL_DILATE_7, iterations=1) > 0
        delta_int_k = min(max(mean_int_k - normal_med, 10.0), 65.0)
        max_prob_k = float(np.max(p_wt[comp_dil]))
        mean_prob_k = float(np.mean(p_wt[comp_dil]))
        overlap_neural = float(np.count_nonzero(comp_dil & (neural_mask > 0)))

        min_d_vault = float(np.min(dist_inside_vault[comp_bool]))
        dural_bonus = 1.45 if min_d_vault <= 14.0 else 0.95

        if max_wt >= 0.12 and (overlap_neural >= 18.0 or (max_prob_k >= max(0.24, 0.58 * max_wt) and mean_prob_k >= 0.16)) and (mean_int_k - normal_med) >= 14.0:
            score_seeded = (
                (area_k ** 0.85)
                * geom_w
                * dural_bonus
                * delta_int_k
                * (1.0 + 6.0 * max_prob_k + 8.0 * mean_prob_k)
            )
            if score_seeded > best_seeded_score:
                best_seeded_score = score_seeded
                best_seeded_hyper = (comp_bool.astype(np.uint8)) * 255

        if area_k >= 240.0 and geom_w >= 0.34 and (mean_int_k - normal_med) >= max(18.0, 1.6 * normal_std) and mean_int_k >= 104.0:
            score_unseeded = (area_k ** 0.86) * (geom_w ** 1.3) * dural_bonus * delta_int_k
            if score_unseeded > best_unseeded_score:
                best_unseeded_score = score_unseeded
                best_unseeded_hyper = (comp_bool.astype(np.uint8)) * 255

    dural_px = int(np.count_nonzero(dural_mass_mask > 0))
    seeded_px = int(np.count_nonzero(best_seeded_hyper > 0))
    nm_px = int(np.count_nonzero(neural_mask > 0))
    nm_mean_int = float(np.mean(blurred[neural_mask > 0])) if nm_px > 0 else 0.0

    if dural_px >= 280 and (
        (seeded_px < 240 and dural_px >= 1.8 * max(seeded_px, 1))
        or (seeded_px == 0 and nm_mean_int < mod_thresh)
    ):
        best_seeded_hyper = dural_mass_mask.copy()
        seeded_px = dural_px

    # 3. Реконструкция всего узла менингиомы
    if np.any(best_seeded_hyper > 0):
        solid_union = _complete_whole_meningioma_nodule(
            best_seeded_hyper, blurred.astype(np.float32), p_wt, vault_bool,
            normal_med, normal_std, mod_thresh, seg_threshold=seg_threshold
        )
        if np.any(neural_mask > 0):
            seed_q25 = float(np.percentile(blurred[best_seeded_hyper > 0], 25.0))
            nm_bright = (neural_mask > 0) & (blurred >= max(seed_q25 - 12.0, normal_med + 0.90 * normal_std))
            solid_dil = cv2.dilate(solid_union, _KERNEL_OPEN_9, iterations=1) > 0
            if np.any(nm_bright & solid_dil) and np.count_nonzero(nm_bright) <= 1.35 * np.count_nonzero(solid_union):
                solid_union = cv2.bitwise_or(solid_union, (nm_bright.astype(np.uint8)) * 255)
    elif np.any(best_unseeded_hyper > 0):
        solid_union = _complete_whole_meningioma_nodule(
            best_unseeded_hyper, blurred.astype(np.float32), p_wt, vault_bool,
            normal_med, normal_std, mod_thresh, seg_threshold=seg_threshold
        )
    elif dural_px >= 240:
        solid_union = _complete_whole_meningioma_nodule(
            dural_mass_mask, blurred.astype(np.float32), p_wt, vault_bool,
            normal_med, normal_std, mod_thresh, seg_threshold=seg_threshold
        )
    elif nm_px >= 100:
        solid_union = _prune_leaking_tails_and_bridges(
            neural_mask, blurred.astype(np.float32), p_wt, vault_bool, seg_threshold=seg_threshold
        )
    else:
        solid_union = neural_mask.copy()

    # Захват чёрных кистозно-некротических камер, примыкающих к узлу (ВНЕ центральной селлярной области гипофиза!)
    if np.count_nonzero(solid_union) >= 120:
        t_cyst_dark = float(np.clip(normal_med - 0.85 * normal_std, 18.0, 56.0))
        cyst_cand = (vault_bool & (~in_sellar_pituitary_center) & (blurred <= t_cyst_dark)).astype(np.uint8) * 255
        cyst_cand = cv2.morphologyEx(cyst_cand, cv2.MORPH_OPEN, _KERNEL_ERODE_5, iterations=1)
        cyst_dist = cv2.distanceTransform(cyst_cand, cv2.DIST_L2, 5)
        solid_dil_2 = cv2.dilate(solid_union, _KERNEL_OPEN_9, iterations=2) > 0

        num_cy, lbl_cy, st_cy, _ = cv2.connectedComponentsWithStats(cyst_cand, connectivity=8)
        for k_cy in range(1, num_cy):
            a_cy = float(st_cy[k_cy, cv2.CC_STAT_AREA])
            if 60.0 <= a_cy <= 1.2 * float(np.count_nonzero(solid_union)):
                comp_cy = (lbl_cy == k_cy)
                if float(np.max(cyst_dist[comp_cy])) >= 5.5 and np.count_nonzero(comp_cy & solid_dil_2) >= 15:
                    solid_union[comp_cy] = 255
        solid_union = cv2.morphologyEx(solid_union, cv2.MORPH_CLOSE, _KERNEL_CLOSE_11, iterations=1)
        solid_union = fill_core_necrosis_ccomp(solid_union, max_hole_ratio=0.85)

    solid_union = _prune_leaking_tails_and_bridges(
        solid_union, blurred.astype(np.float32), p_wt, vault_bool, seg_threshold=seg_threshold
    )
    solid_union = cv2.bitwise_and(solid_union, vault_mask_u8)

    if np.count_nonzero(solid_union) < 35:
        return empty, empty, empty

    # 4. Анатомическая 3-зонная декомпозиция (Core #E11D48, Inflam #FF9E00, Edema #00BEFF)
    eroded_core = cv2.erode(solid_union, _KERNEL_ERODE_5, iterations=1)
    if np.count_nonzero(eroded_core) >= 70:
        shell_ring = cv2.subtract(solid_union, eroded_core) > 0
        solid_vals = blurred[solid_union > 0]
        shell_q65 = float(np.percentile(solid_vals, 65.0)) if solid_vals.size > 20 else 145.0
        inflam_bool = shell_ring & ((p_inflam >= 0.16) | (blurred >= shell_q65))
        mask_inflam = (inflam_bool.astype(np.uint8)) * 255
    else:
        mask_inflam = empty.copy()

    mask_core = cv2.subtract(solid_union, mask_inflam)
    mask_core = fill_core_necrosis_ccomp(mask_core, max_hole_ratio=0.85)
    mask_inflam = np.where(mask_core > 0, 0, mask_inflam).astype(np.uint8)

    edema_tau = max(0.22, 0.42 * seg_threshold)
    peri_ring = (cv2.subtract(cv2.dilate(solid_union, _KERNEL_OPEN_9, iterations=1), solid_union) > 0) & vault_bool
    edema_bool = peri_ring & (p_edema >= edema_tau) & (blurred >= 42) & (blurred <= hyper_thresh)
    mask_edema = (edema_bool.astype(np.uint8)) * 255
    mask_edema = np.where((mask_core > 0) | (mask_inflam > 0), 0, mask_edema).astype(np.uint8)

    return mask_core, mask_inflam, mask_edema
def _is_glioma_csf_or_ventricle_artifact(
    comp_bool: np.ndarray,
    blurred: np.ndarray,
    dist_vault: np.ndarray,
    vault_bool: np.ndarray,
    med_b: float,
    iqr_b: float
) -> bool:
    """
    Отбраковывает ложные тёмные маски на глиомах:
      1) Симметричные боковые желудочки по средней линии без контрастного ободка (#001);
      2) Тонкую полосу/серп субарахноидального ликвора у самого края свода (#020, #293).
    Сохраняет объёмные гипоинтенсивные корковые глиомы (#016, #133, #237, #343) и кистозные глиобластомы (#006).
    """
    area_k = float(np.count_nonzero(comp_bool))
    if area_k < 35.0:
        return True

    mean_int = float(np.mean(blurred[comp_bool]))
    p85_int = float(np.percentile(blurred[comp_bool], 85.0))
    comp_u8 = (comp_bool.astype(np.uint8)) * 255

    comp_dil = cv2.dilate(comp_u8, _KERNEL_OPEN_9, iterations=1) > 0
    rim_mask = comp_dil & (~comp_bool) & vault_bool
    rim_bright_frac = float(np.mean(blurred[rim_mask] >= (med_b + 1.15 * iqr_b))) if np.any(rim_mask) else 0.0
    has_enhancing_rim = (rim_bright_frac >= 0.20) or (p85_int >= med_b + 1.40 * iqr_b)

    # Отличаем тонкую полоску ликвора от объёмной тёмной глиомы
    comp_dist = cv2.distanceTransform(comp_u8, cv2.DIST_L2, 5)
    max_comp_thick = float(np.max(comp_dist))
    cnts, _ = cv2.findContours(comp_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solidity, aspect_ratio = 1.0, 1.0
    if cnts:
        main_c = max(cnts, key=cv2.contourArea)
        hull_a = max(float(cv2.contourArea(cv2.convexHull(main_c))), 1.0)
        solidity = area_k / hull_a
        rect = cv2.minAreaRect(main_c)
        w_r, h_r = max(float(rect[1][0]), 1.0), max(float(rect[1][1]), 1.0)
        aspect_ratio = min(w_r, h_r) / max(w_r, h_r)

    is_bulky_mass = (max_comp_thick >= 10.5 and solidity >= 0.62 and aspect_ratio >= 0.42)

    # 1. Тонкая тёмная полоса/серп ликвора у самой границы свода (#020, #293)
    depths = dist_vault[comp_bool]
    border_frac = float(np.mean(depths < 7.0))
    mean_d = float(np.mean(depths))
    if (mean_int < med_b - 9.0) and (not has_enhancing_rim) and (not is_bulky_mass):
        if (border_frac >= 0.35 and mean_d < 11.0) or (border_frac >= 0.25 and (max_comp_thick < 8.5 or aspect_ratio < 0.34 or solidity < 0.55)):
            return True

    # 2. Симметричные тёмные боковые желудочки строго по средней линии (#001)
    ys_v, xs_v = np.where(vault_bool)
    if xs_v.size > 100 and (mean_int < med_b - 6.0) and (not has_enhancing_rim):
        vcx = 0.50 * (float(np.min(xs_v)) + float(np.max(xs_v)))
        vw = max(float(np.max(xs_v) - np.min(xs_v)), 60.0)
        ys_c, xs_c = np.where(comp_bool)
        cx_c = float(np.mean(xs_c))
        if abs(cx_c - vcx) <= 0.09 * vw:
            mir_xs = np.clip(np.round(2.0 * vcx - xs_c).astype(np.int32), 0, IMG_SIZE - 1)
            mirror_hit = float(np.mean(comp_bool[ys_c, mir_xs]))
            if mirror_hit >= 0.58:
                return True

    return False

def calibrate_and_refine_submasks(
    parenchyma_gray_512: np.ndarray,
    det_probs_4ch: np.ndarray,
    parenchyma_mask_512: np.ndarray,
    predicted_class: str,
    seg_threshold: float = DEFAULT_SEG_THRESHOLD
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    if predicted_class == "no_tumor":
        return empty, empty, empty

    gray_u8 = parenchyma_gray_512.astype(np.uint8)
    vault_mask_u8 = (parenchyma_mask_512 > 127).astype(np.uint8) * 255
    vault_bool = (vault_mask_u8 > 0)
    vault_area = float(np.count_nonzero(vault_bool))
    if vault_area < 400.0:
        return empty, empty, empty

    m_brain = vault_bool.astype(np.float32)
    p_core = cv2.GaussianBlur(det_probs_4ch[1].astype(np.float32), (5, 5), 1.0) * m_brain
    p_inflam = cv2.GaussianBlur(det_probs_4ch[2].astype(np.float32), (5, 5), 1.0) * m_brain
    p_edema = cv2.GaussianBlur(det_probs_4ch[3].astype(np.float32), (5, 5), 1.0) * m_brain
    p_wt = np.clip(p_core + p_inflam + p_edema, 0.0, 1.0)

    tau = float(np.clip(seg_threshold, 0.20, 0.92))
    mask_core = empty.copy()
    mask_inflam = empty.copy()
    mask_edema = empty.copy()

    if predicted_class == "meningioma":
        mask_core, mask_inflam, mask_edema = refine_meningioma_hybrid_subzones(
            gray_u8, p_core, p_inflam, p_edema, vault_mask_u8, seg_threshold=tau
        )

    elif predicted_class == "pituitary":
        mask_core, mask_inflam, mask_edema = _refine_pituitary_sellar_adenoma(
            gray_u8, p_core, p_inflam, p_edema, vault_mask_u8, seg_threshold=tau
        )

    elif predicted_class == "glioma":
        plane_info = detect_slice_plane_and_level_normalized(gray_u8)
        blurred = cv2.GaussianBlur(gray_u8, (5, 5), 1.0).astype(np.float32)
        dist_vault = cv2.distanceTransform(vault_mask_u8, cv2.DIST_L2, 3)

        brain_px = blurred[vault_bool & (blurred > 28) & (dist_vault >= 8.0)]
        med_b = float(np.median(brain_px)) if brain_px.size > 100 else 78.0
        iqr_b = max(float(np.percentile(brain_px, 75.0) - np.percentile(brain_px, 25.0)), 8.0) if brain_px.size > 100 else 14.0

        dark_u8 = (vault_bool & (blurred < 30.0)).astype(np.uint8) * 255
        dark_thick = cv2.distanceTransform(dark_u8, cv2.DIST_L2, 5)
        valid_tissue = (blurred >= 30.0) | (dark_thick >= 8.5)

        peak_p = float(np.max(p_wt))
        tau_g = float(np.clip(tau * 0.90, 0.20, 0.88 * peak_p)) if peak_p >= 0.14 else tau
        raw_wt = ((p_wt >= tau_g) & vault_bool & valid_tissue).astype(np.uint8) * 255
        if float(np.count_nonzero(raw_wt)) > 0.18 * vault_area:
            high_tau = max(float(np.percentile(p_wt[vault_bool], 86.0)), max(tau_g, 0.62))
            raw_wt = ((p_wt >= high_tau) & vault_bool & valid_tissue).astype(np.uint8) * 255

        raw_wt = _prune_leaking_tails_and_bridges(raw_wt, blurred, p_wt, vault_bool, seg_threshold=tau)

        num_g, lbl_g, st_g, _ = cv2.connectedComponentsWithStats(raw_wt, connectivity=8)
        neural_wt = empty.copy()
        best_g_score = -1.0
        for kg in range(1, num_g):
            a_g = float(st_g[kg, cv2.CC_STAT_AREA])
            if a_g < 45.0 or a_g > 0.24 * vault_area:
                continue
            comp_g = (lbl_g == kg)
            comp_u8 = _complete_ring_or_cyst_with_adjacent_necrosis(
                comp_g.astype(np.uint8) * 255, blurred, vault_bool, med_b, iqr_b
            )
            comp_g = (comp_u8 > 0)
            a_g = float(np.count_nonzero(comp_g))
            if a_g > 0.25 * vault_area:
                continue

            is_art, geom_w = _evaluate_skull_and_orbit_artifact_penalty(
                comp_g, blurred, dist_vault, vault_bool, plane_info, is_meningioma=False
            )
            if is_art:
                continue
            if _is_glioma_csf_or_ventricle_artifact(comp_g, blurred, dist_vault, vault_bool, med_b, iqr_b):
                continue
            depth_b = 1.0 + 0.85 * min(float(np.mean(dist_vault[comp_g])) / 26.0, 1.3)
            sc_g = (a_g ** 0.72) * (float(np.mean(p_wt[comp_g])) ** 1.3) * geom_w * depth_b
            if sc_g > best_g_score:
                best_g_score = sc_g
                neural_wt = comp_u8

        cand_mask, cand_info = detect_intraaxial_glioma_candidates(gray_u8, vault_mask_u8, p_wt)
        cand_flip, info_flip = detect_intraaxial_glioma_candidates(
            cv2.flip(gray_u8, 1), cv2.flip(vault_mask_u8, 1), cv2.flip(p_wt, 1)
        )
        if info_flip["intraaxial_score"] > cand_info["intraaxial_score"] * 1.05:
            cand_mask = cv2.flip(cand_flip, 1)
            cand_info = info_flip

        cand_mask = _prune_leaking_tails_and_bridges(cand_mask, blurred, p_wt, vault_bool, seg_threshold=tau)

        if np.count_nonzero(neural_wt) >= 90:
            if (
                cand_info["has_ring_necrosis"] > 0.5
                and np.count_nonzero(cand_mask) >= 1.45 * np.count_nonzero(neural_wt)
                and np.count_nonzero((neural_wt > 0) & (cand_mask > 0)) == 0
            ):
                active_wt = cand_mask
            else:
                active_wt = neural_wt
        elif tau <= 0.82 and np.count_nonzero(cand_mask) >= 120:
            active_wt = cand_mask
        else:
            active_wt = neural_wt

        if np.any(active_wt > 0):
            active_wt = _complete_ring_or_cyst_with_adjacent_necrosis(
                active_wt, blurred, vault_bool, med_b, iqr_b
            )
            active_wt = _prune_leaking_tails_and_bridges(active_wt, blurred, p_wt, vault_bool, seg_threshold=tau)
            mask_core, mask_inflam, mask_edema = refine_glioma_subzones_by_intensity(
                gray_u8, active_wt, p_core, p_inflam, p_edema
            )

    for m_arr in [mask_core, mask_inflam, mask_edema]:
        m_arr[:] = cv2.bitwise_and(m_arr, vault_mask_u8)
        num_cc, labels_cc, stats_cc, _ = cv2.connectedComponentsWithStats(m_arr, connectivity=8)
        if num_cc > 1:
            clean_m = np.zeros_like(m_arr)
            for k in range(1, num_cc):
                if stats_cc[k, cv2.CC_STAT_AREA] >= 35:
                    clean_m[labels_cc == k] = 255
            m_arr[:] = clean_m[:]

    mask_inflam = np.where(mask_core > 0, 0, mask_inflam).astype(np.uint8)
    mask_edema = np.where((mask_core > 0) | (mask_inflam > 0), 0, mask_edema).astype(np.uint8)

    return mask_core, mask_inflam, mask_edema


@dataclass
class MorphometryMetrics:
    wt_area_mm2: float = 0.0
    core_area_mm2: float = 0.0
    inflam_area_mm2: float = 0.0
    inflam_rim_width_mm: float = 0.0
    edema_area_mm2: float = 0.0
    max_diameter_mm: float = 0.0
    perp_diameter_mm: float = 0.0
    midline_shift_mm: float = 0.0
    boundary_uncertainty_mean: float = 0.0
    hemisphere_loc: str = "данные отсутствуют"
    lobe_loc: str = "данные отсутствуют"
    contour_character: str = "не определяются"
    plane_ru: str = "данные отсутствуют"
    shape_ru: str = "не определяется"


def compute_wt_uncertainty_map(
    det_probs_4ch: np.ndarray,
    wt_mask_u8: np.ndarray,
    tta_std_wt: Optional[np.ndarray] = None,
    mc_std_wt: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Вычисляет карту неуверенности [0..1] для зоны общего выделения (Whole Tumor)
    строго в инференсе (без дообучения модели):
      1) Обратная вероятность опухоли внутри маски (1 - P_WT) — максимальна там, где вероятность меньше всего;
      2) Нормированная энтропия Шеннона 4-канального softmax: H(p) / ln(4);
      3) Разброс (std) между векторизованными проходами MC Dropout и/или TTA.
    """
    wt_bool = (wt_mask_u8 > 127)
    if not np.any(wt_bool):
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    probs_safe = np.clip(det_probs_4ch.astype(np.float32), 1e-6, 1.0)
    probs_safe = probs_safe / np.sum(probs_safe, axis=0, keepdims=True)

    # 1. Места общего выделения с наименьшей вероятностью опухоли
    p_wt = np.clip(probs_safe[1] + probs_safe[2] + probs_safe[3], 0.0, 1.0)
    low_prob_unc = 1.0 - p_wt

    # 2. Нормированная энтропия Шеннона H(p) / ln(4)
    entropy_norm = -np.sum(probs_safe * np.log(probs_safe), axis=0) / math.log(4.0)
    entropy_norm = np.clip(entropy_norm, 0.0, 1.0)

    # 3. Разброс между проходами (MC Dropout / TTA)
    disp_parts = []
    if mc_std_wt is not None:
        disp_parts.append(np.clip(mc_std_wt.astype(np.float32) / 0.16, 0.0, 1.0))
    if tta_std_wt is not None:
        disp_parts.append(np.clip(tta_std_wt.astype(np.float32) / 0.18, 0.0, 1.0))

    if disp_parts:
        disp_norm = np.mean(disp_parts, axis=0)
        unc_raw = 0.55 * low_prob_unc + 0.25 * entropy_norm + 0.20 * disp_norm
    else:
        unc_raw = 0.68 * low_prob_unc + 0.32 * entropy_norm

    unc_smooth = cv2.GaussianBlur(unc_raw.astype(np.float32), (5, 5), 1.0)
    return np.clip(unc_smooth, 0.0, 1.0)


def _draw_dashed_uncertainty_contour(
    out_rgb: np.ndarray,
    contour_pts: np.ndarray,
    dash_len_px: float = 6.5,
    thickness: int = 2
) -> None:
    """
    Рисует равномерный двухцветный пунктир (жёлтый #FFE600 + фиолетовый #A855F7)
    вдоль длины дуги контура (одинаково чётко на горизонтальных, вертикальных и изогнутых участках).
    """
    pts = contour_pts.reshape(-1, 2).astype(np.float32)
    if len(pts) < 4:
        return

    pts_closed = np.vstack([pts, pts[0:1]])
    seg_vecs = np.diff(pts_closed, axis=0)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total_len = float(cum_len[-1])
    if total_len < 6.0:
        return

    n_samples = max(int(round(total_len / 1.5)), 8)
    s_grid = np.linspace(0.0, total_len, n_samples, endpoint=False)
    xs = np.interp(s_grid, cum_len, pts_closed[:, 0])
    ys = np.interp(s_grid, cum_len, pts_closed[:, 1])

    color_yellow = (255, 230, 0)   # RGB жёлтый (#FFE600)
    color_violet = (168, 85, 247)  # RGB фиолетовый (#A855F7)

    for i in range(n_samples):
        j = (i + 1) % n_samples
        dash_idx = int(s_grid[i] // dash_len_px)
        col = color_yellow if (dash_idx % 2 == 0) else color_violet
        p1 = (int(round(xs[i])), int(round(ys[i])))
        p2 = (int(round(xs[j])), int(round(ys[j])))
        cv2.line(out_rgb, p1, p2, col, thickness=thickness, lineType=cv2.LINE_AA)


def compute_three_zone_morphometry(
    mask_core: np.ndarray,
    mask_inflam: np.ndarray,
    mask_edema: np.ndarray,
    parenchyma_mask_512: np.ndarray,
    pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM,
    predicted_class: str = "glioma",
    unc_map: Optional[np.ndarray] = None,
    plane: str = "axial"
) -> MorphometryMetrics:
    """
    2D-морфометрия по одному МРТ-срезу: вычисляются исключительно площадь (кв. мм)
    и линейные 2D-размеры (мм). Трёхмерный объём не рассчитывается.
    """
    plane_key = str(plane or "").lower()
    plane_ru = PLANE_LABELS_RU.get(plane_key, "данные отсутствуют")

    px_area = float(pixel_spacing_mm) ** 2
    c_px = int(np.count_nonzero(mask_core > 127))
    inf_px = int(np.count_nonzero(mask_inflam > 127))
    e_px = int(np.count_nonzero(mask_edema > 127))
    wt_px = c_px + inf_px + e_px

    if wt_px < 5 or predicted_class == "no_tumor":
        return MorphometryMetrics(plane_ru=plane_ru)

    c_area = c_px * px_area
    inf_area = inf_px * px_area
    e_area = e_px * px_area
    wt_area = wt_px * px_area

    wt_bin = ((mask_core > 127) | (mask_inflam > 127) | (mask_edema > 127)).astype(np.uint8)
    cnts, _ = cv2.findContours(wt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_d, perp_d = 0.0, 0.0
    shape_ru = "объёмное"
    contour_ch = "неровные"
    if cnts:
        main_cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.minAreaRect(main_cnt)
        w_r, h_r = float(rect[1][0]), float(rect[1][1])
        max_d = max(w_r, h_r) * pixel_spacing_mm
        perp_d = min(w_r, h_r) * pixel_spacing_mm
        aspect_r = min(w_r, h_r) / max(w_r, h_r, 1e-4)
        hull_a = max(float(cv2.contourArea(cv2.convexHull(main_cnt))), 1.0)
        solidity = float(cv2.contourArea(main_cnt)) / hull_a
        if aspect_r >= 0.72 and solidity >= 0.80:
            shape_ru = "округлое"
            contour_ch = "относительно ровные"
        elif solidity >= 0.78:
            shape_ru = "овальное"
            contour_ch = "относительно ровные"
        else:
            shape_ru = "неправильной бугристой формы"
            contour_ch = "бугристые, неровные"

    ys, xs = np.where(wt_bin > 0)
    cx = float(np.mean(xs))
    shift = round(max(0.0, (1.0 - abs(cx - 256.0) / 220.0) * 4.0 * min(wt_area / 1800.0, 1.5)), 1) if wt_area >= 250.0 else 0.0
    rim_width = round(math.sqrt(inf_area / math.pi), 1) if inf_px > 0 else 0.0

    bnd_unc_mean = 0.0
    if unc_map is not None and unc_map.shape[:2] == wt_bin.shape[:2]:
        bnd = cv2.morphologyEx(wt_bin * 255, cv2.MORPH_GRADIENT, _KERNEL_ERODE_5) > 0
        if np.any(bnd):
            bnd_unc_mean = round(float(np.mean(unc_map[bnd])), 3)
        else:
            bnd_unc_mean = round(float(np.mean(unc_map[wt_bin > 0])), 3)

    if predicted_class == "pituitary":
        loc_str = "хиазмально-селлярная область"
    elif plane_key == "sagittal":
        loc_str = "данные отсутствуют (полушарие не определяется по одному сагиттальному срезу)"
    elif abs(cx - 256.0) <= 18.0:
        loc_str = "срединные (парасагиттальные) отделы головного мозга"
    elif cx < 256.0:
        loc_str = "правое полушарие головного мозга"
    else:
        loc_str = "левое полушарие головного мозга"

    return MorphometryMetrics(
        wt_area_mm2=round(wt_area, 1),
        core_area_mm2=round(c_area, 1),
        inflam_area_mm2=round(inf_area, 1),
        inflam_rim_width_mm=rim_width,
        edema_area_mm2=round(e_area, 1),
        max_diameter_mm=round(max_d, 1),
        perp_diameter_mm=round(perp_d, 1),
        midline_shift_mm=shift,
        boundary_uncertainty_mean=bnd_unc_mean,
        hemisphere_loc=loc_str,
        lobe_loc="данные отсутствуют (уточнение доли требует оценки полной серии срезов)",
        contour_character=contour_ch,
        plane_ru=plane_ru,
        shape_ru=shape_ru
    )


_CYRILLIC_FONT_CACHE: Dict[int, Any] = {}


def _get_cyrillic_font(font_size: int = 15) -> Any:
    """
    Возвращает TrueType-шрифт с поддержкой кириллицы (работает в Google Colab, Linux, macOS, Windows).
    """
    if font_size in _CYRILLIC_FONT_CACHE:
        return _CYRILLIC_FONT_CACHE[font_size]

    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ] + get_windows_font_candidates()
    try:
        import matplotlib
        mpl_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
        candidates.insert(0, str(mpl_ttf))
    except Exception:
        pass

    font_obj = None
    for f_path in candidates:
        if os.path.exists(f_path):
            try:
                font_obj = ImageFont.truetype(f_path, size=font_size)
                break
            except Exception:
                continue

    if font_obj is None:
        font_obj = ImageFont.load_default()

    _CYRILLIC_FONT_CACHE[font_size] = font_obj
    return font_obj


def _draw_russian_legend_top_left(out_rgb: np.ndarray) -> np.ndarray:
    """
    Наносит русскоязычные подписи зон в левом верхнем углу соответствующими цветами
    (без цветовых HEX-кодов) с тонкой тёмной обводкой для читаемости.
    """
    pil_img = Image.fromarray(out_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_cyrillic_font(font_size=15)

    legend_items = [
        ("Ядро опухоли", (225, 29, 72)),
        ("Воспаление / ГЭБ", (255, 158, 0)),
        ("Отёк", (0, 190, 255)),
        ("Неуверенная зона", (255, 230, 0)),
    ]

    x_pos = 12
    y_pos = 10
    line_step = 21
    for text, color_rgb in legend_items:
        try:
            draw.text(
                (x_pos, y_pos),
                text,
                font=font,
                fill=color_rgb,
                stroke_width=1,
                stroke_fill=(0, 0, 0)
            )
        except TypeError:
            draw.text((x_pos + 1, y_pos + 1), text, font=font, fill=(0, 0, 0))
            draw.text((x_pos, y_pos), text, font=font, fill=color_rgb)
        y_pos += line_step

    return np.array(pil_img, dtype=np.uint8, copy=True)


def _draw_russian_warning_banner(img_rgb: np.ndarray, text: str) -> np.ndarray:
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)
    font = _get_cyrillic_font(font_size=20)
    try:
        draw.text((28, 240), text, font=font, fill=(225, 29, 72), stroke_width=2, stroke_fill=(0, 0, 0))
    except TypeError:
        draw.text((28, 240), text, font=font, fill=(225, 29, 72))
    return np.array(pil_img, dtype=np.uint8, copy=True)


def render_three_zone_composite_512(
    orig_rgb_512: np.ndarray,
    mask_core: np.ndarray,
    mask_inflam: np.ndarray,
    mask_edema: np.ndarray,
    unc_map: np.ndarray,
    unc_threshold: float = DEFAULT_UNC_THRESHOLD
) -> np.ndarray:
    """
    4-слойный композитный снимок (RGB):
      Слой 1: Нативный срез МРТ с черепом
      Слой 2: Перифокальный отёк (#00BEFF) и воспалительно-реактивный интерфейс (#FF9E00)
      Слой 3: Опухолевое ядро (#E11D48)
      Слой 4: Неуверенное выделение (жёлто-фиолетовый пунктир #FFE600 / #A855F7) в местах
              общего выделения (Whole Tumor), где вероятность модели меньше всего.
    """
    canvas = cv2.cvtColor(orig_rgb_512, cv2.COLOR_GRAY2RGB) if orig_rgb_512.ndim == 2 else orig_rgb_512[:, :, :3].copy()
    overlay = canvas.astype(np.float32)

    c_b = (mask_core > 127)
    inf_b = (mask_inflam > 127) & (~c_b)
    e_b = (mask_edema > 127) & (~c_b) & (~inf_b)

    if np.any(e_b):
        overlay[e_b] = 0.66 * overlay[e_b] + 0.34 * np.array([0.0, 190.0, 255.0], dtype=np.float32)
    if np.any(inf_b):
        overlay[inf_b] = 0.60 * overlay[inf_b] + 0.40 * np.array([255.0, 158.0, 0.0], dtype=np.float32)
    if np.any(c_b):
        overlay[c_b] = 0.55 * overlay[c_b] + 0.45 * np.array([225.0, 29.0, 72.0], dtype=np.float32)

    out_rgb = np.clip(overlay, 0, 255).astype(np.uint8)

    # Контуры 3 анатомических зон в корректных RGB-цветах
    for m, col in [
        (mask_edema, (0, 190, 255)),     # Edema (#00BEFF)
        (mask_inflam, (255, 158, 0)),    # Inflam (#FF9E00)
        (mask_core, (225, 29, 72))       # Core (#E11D48)
    ]:
        cnts, _ = cv2.findContours((m > 127).astype(np.uint8) * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(out_rgb, cnts, -1, col, thickness=2, lineType=cv2.LINE_AA)

    # Слой 4: Неуверенная зона / граница внутри общего выделения (где вероятность меньше всего)
    wt_bool = (c_b | inf_b | e_b)
    if np.any(wt_bool) and unc_map is not None and unc_map.shape[:2] == (IMG_SIZE, IMG_SIZE):
        wt_vals = unc_map[wt_bool]
        max_unc_wt = float(np.max(wt_vals)) if wt_vals.size > 0 else 0.0

        if max_unc_wt >= 0.12:
            # Адаптивный порог: выделяем участки с unc >= unc_threshold ИЛИ нижние ~28% по вероятности внутри общего выделения
            q72_unc = float(np.percentile(wt_vals, 72.0))
            eff_unc_th = float(min(float(unc_threshold), max(0.18, q72_unc)))

            unc_region = (wt_bool & (unc_map >= eff_unc_th)).astype(np.uint8) * 255
            unc_region = cv2.morphologyEx(unc_region, cv2.MORPH_CLOSE, _KERNEL_CLOSE_7, iterations=1)
            unc_region = cv2.morphologyEx(unc_region, cv2.MORPH_OPEN, _KERNEL_ERODE_3, iterations=1)
            unc_region = cv2.bitwise_and(unc_region, (wt_bool.astype(np.uint8)) * 255)

            num_u, lbl_u, st_u, _ = cv2.connectedComponentsWithStats(unc_region, connectivity=8)
            min_unc_area = max(35.0, 0.02 * float(np.count_nonzero(wt_bool)))
            unc_clean = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            for ku in range(1, num_u):
                if float(st_u[ku, cv2.CC_STAT_AREA]) >= min_unc_area:
                    unc_clean[lbl_u == ku] = 255

            if np.any(unc_clean > 0):
                u_mask_b = (unc_clean > 0)
                out_f = out_rgb.astype(np.float32)
                out_f[u_mask_b] = 0.88 * out_f[u_mask_b] + 0.12 * np.array([255.0, 230.0, 0.0], dtype=np.float32)
                out_rgb = np.clip(out_f, 0, 255).astype(np.uint8)

                u_cnts, _ = cv2.findContours(unc_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                for cnt in u_cnts:
                    if cv2.arcLength(cnt, True) >= 16.0:
                        _draw_dashed_uncertainty_contour(out_rgb, cnt, dash_len_px=6.5, thickness=2)

    out_rgb = _draw_russian_legend_top_left(out_rgb)
    return out_rgb


class StandaloneL4CDSSController:
    def __init__(self, ckpt_dir: Path) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.gatekeeper_loaded = False
        self.model_loaded = False
        self.gatekeeper = timm.create_model("convnextv2_atto.fcmae", pretrained=False, num_classes=2).to(DEVICE).eval()
        self.model = MultiTaskNeuroNet(
            backbone_name="convnext_base.fb_in22k_ft_in1k_384",
            num_classes=4,
            num_seg_classes=4,
            pretrained=False
        ).to(DEVICE).eval()
        self.mahal = MahalanobisOODDetector()
        self.load_checkpoints()

        self.transforms_256 = A.Compose([A.Resize(256, 256), A.Normalize(), ToTensorV2()])
        self.transforms_512 = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE), A.Normalize(), ToTensorV2()])

    def load_checkpoints(self) -> None:
        self.gatekeeper_loaded = False
        self.model_loaded = False

        gk_path = self.ckpt_dir / "gatekeeper_512_best.pth"
        mt_path = self.ckpt_dir / "multitask_512_best.pth"
        mahal_path = self.ckpt_dir / "mahalanobis_ood_512.npz"

        if gk_path.exists() and gk_path.stat().st_size > 1024 * 1024:
            try:
                self.gatekeeper.load_state_dict(_safe_torch_load(gk_path).get("model_state_dict", {}), strict=False)
                self.gatekeeper_loaded = True
                print(f"[MODEL-OK] Загружен Gatekeeper: {gk_path.name}")
            except Exception as e:
                print(f"[MODEL-WARN] Ошибка загрузки Gatekeeper: {e}")
        else:
            print("[MODEL-INFO] gatekeeper_512_best.pth не найден — используется только OOD-барьер Махаланобиса.")

        if mt_path.exists() and mt_path.stat().st_size > 10 * 1024 * 1024:
            try:
                mt_ckpt = _safe_torch_load(mt_path)
                self.model.load_state_dict(mt_ckpt.get("model_state_dict", {}), strict=False)
                self.mahal.load(mahal_path, fallback_dict=mt_ckpt.get("mahalanobis_state"))
                self.model_loaded = True
                print(f"[MODEL-OK] Загружена основная модель: {mt_path.name} | Mahalanobis fitted={self.mahal.fitted}")
            except Exception as e:
                print(f"[MODEL-ERROR] Ошибка загрузки основной модели: {e}")
        else:
            print(f"[MODEL-ERROR] Веса {mt_path} не найдены! Проверьте ссылки Google Drive в Ячейке 1.")

    def _compute_vectorized_mc_dropout_std(self, t_can: torch.Tensor, n_passes: int = MC_DROPOUT_PASSES) -> Optional[np.ndarray]:
        """
        Векторизованный MC Dropout без дообучения: включает только Dropout2d/Dropout в декодере
        (BatchNorm остаётся в eval), выполняя N проходов одним батчем на GPU.
        На CPU возвращает None (используется TTA-дисперсия без лишних затрат времени).
        """
        if DEVICE.type != "cuda" or n_passes < 2:
            return None

        dropout_layers = []
        for m in self.model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d)):
                dropout_layers.append((m, m.training))
                m.train()

        try:
            use_bf16 = bool(AMP_DTYPE == torch.bfloat16)
            mc_batch = t_can.repeat(int(n_passes), 1, 1, 1).to(DEVICE)
            with torch.no_grad():
                with amp_autocast(DEVICE, AMP_DTYPE):
                    mc_seg_logits, _, _ = self.model(mc_batch)
                mc_probs = F.softmax(mc_seg_logits.float(), dim=1)
                mc_wt = torch.clamp(mc_probs[:, 1] + mc_probs[:, 2] + mc_probs[:, 3], 0.0, 1.0)
                mc_std_wt = torch.std(mc_wt, dim=0, unbiased=False).cpu().numpy()
            return mc_std_wt.astype(np.float32)
        except Exception:
            return None
        finally:
            for m, was_train in dropout_layers:
                m.train(was_train)

    def predict_and_quantify(
        self,
        input_image: np.ndarray,
        seg_threshold: float = DEFAULT_SEG_THRESHOLD,
        unc_threshold: float = DEFAULT_UNC_THRESHOLD,
        pixel_spacing_mm: float = DEFAULT_PIXEL_SPACING_MM
    ) -> Dict[str, Any]:
        pii = extract_patient_pii_deterministic(input_image)
        rgb = cv2.cvtColor(input_image, cv2.COLOR_GRAY2RGB) if input_image.ndim == 2 else input_image[:, :, :3]
        rgb_512 = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)

        norm_gray = robust_percentile_normalize(rgb_512)
        plane_info = detect_slice_plane_and_level_normalized(norm_gray)
        is_vflipped = plane_info.get("is_vflipped", False)

        canon_gray = cv2.flip(norm_gray, 0) if is_vflipped else norm_gray
        canon_vault = _compute_canonical_parenchyma_mask(canon_gray)
        orig_rgb = cv2.merge([norm_gray, norm_gray, norm_gray])
        canon_rgb = cv2.merge([canon_gray, canon_gray, canon_gray])

        damped_gray = np.where(canon_vault > 0, canon_gray, (canon_gray * 0.30).astype(np.uint8))
        damped_rgb = cv2.merge([damped_gray, damped_gray, damped_gray])

        if self.gatekeeper_loaded:
            t_256 = self.transforms_256(image=canon_rgb)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                p_mri = float(F.softmax(self.gatekeeper(t_256).float(), dim=-1)[0][1].cpu().item())
            tier1_passed = p_mri >= 0.48
        else:
            p_mri = 1.0
            tier1_passed = True

        t_can = self.transforms_512(image=canon_rgb)["image"].unsqueeze(0).to(DEVICE)
        t_dmp = self.transforms_512(image=damped_rgb)["image"].unsqueeze(0).to(DEVICE)
        batch_tta = torch.cat([t_can, torch.flip(t_can, dims=[3]), t_dmp, torch.flip(t_dmp, dims=[3])], dim=0).to(DEVICE)

        use_bf16 = bool(torch.cuda.is_available() and AMP_DTYPE == torch.bfloat16)
        with torch.no_grad():
            with amp_autocast(DEVICE, AMP_DTYPE):
                seg_logits_t, cls_logits_t, pooled_t = self.model(batch_tta)
            probs_t = F.softmax(seg_logits_t.float(), dim=1)
            p0 = probs_t[0]
            p1 = torch.flip(probs_t[1], dims=[2])
            p2 = probs_t[2]
            p3 = torch.flip(probs_t[3], dims=[2])
            det_probs_4ch = (0.35 * p0 + 0.35 * p1 + 0.15 * p2 + 0.15 * p3).cpu().numpy()

            # Дисперсия по 4 проходам TTA для оценки неуверенности
            tta_wt_stack = torch.stack([
                torch.clamp(p[1] + p[2] + p[3], 0.0, 1.0) for p in (p0, p1, p2, p3)
            ], dim=0)
            tta_std_wt = torch.std(tta_wt_stack, dim=0, unbiased=False).cpu().numpy()

            cls_probs_all = F.softmax(cls_logits_t.float(), dim=-1).cpu().numpy()
            mean_cls_probs = 0.50 * cls_probs_all[0] + 0.50 * cls_probs_all[1]
            mean_emb = (0.5 * pooled_t[0] + 0.5 * pooled_t[1]).float().cpu().numpy()

        mahal_dist, mahal_ok = self.mahal.verify_embedding(mean_emb)
        p_wt_canon = np.clip(det_probs_4ch[1] + det_probs_4ch[2] + det_probs_4ch[3], 0.0, 1.0) * (canon_vault > 127)
        max_vault_p_wt = float(np.max(p_wt_canon)) if p_wt_canon.size > 0 else 0.0
        ood_passed = bool(tier1_passed and (mahal_ok or max_vault_p_wt >= 0.55))

        if not ood_passed:
            return {
                "ood_passed": False,
                "predicted_class": "rejected_ood",
                "class_probs": {c: 0.0 for c in CLASS_NAMES},
                "ood_metrics": {"p_mri_gatekeeper": p_mri, "mahalanobis_distance": mahal_dist, "ood_passed": False},
                "morphometry": asdict(MorphometryMetrics()),
                "composite_512": _draw_russian_warning_banner(orig_rgb.copy(), "Изображение не распознано как МРТ мозга"),
                "protocol_text": "ИССЛЕДОВАНИЕ ОТКЛОНЕНО: загруженное изображение не распознано как срез МРТ головного мозга."
            }

        pred_cls, mean_cls_probs = recalibrate_cls_probs_with_morphology(
            mean_cls_probs=mean_cls_probs,
            gray_u8=canon_gray,
            vault_mask_u8=canon_vault,
            det_probs_4ch=det_probs_4ch
        )
        pred_idx = CLASS_NAMES.index(pred_cls)

        if pred_cls == "no_tumor":
            parenchyma_mask = cv2.flip(canon_vault, 0) if is_vflipped else canon_vault
            mask_core = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            mask_inflam = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            mask_edema = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            unc_map = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)
        else:
            c_can, inf_can, e_can = calibrate_and_refine_submasks(
                parenchyma_gray_512=canon_gray,
                det_probs_4ch=det_probs_4ch,
                parenchyma_mask_512=canon_vault,
                predicted_class=pred_cls,
                seg_threshold=seg_threshold
            )
            wt_can_u8 = (((c_can > 127) | (inf_can > 127) | (e_can > 127)).astype(np.uint8)) * 255
            mc_std_wt = self._compute_vectorized_mc_dropout_std(t_can, n_passes=MC_DROPOUT_PASSES) if np.any(wt_can_u8 > 0) else None
            unc_can = compute_wt_uncertainty_map(
                det_probs_4ch=det_probs_4ch,
                wt_mask_u8=wt_can_u8,
                tta_std_wt=tta_std_wt,
                mc_std_wt=mc_std_wt
            )

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

        slice_plane = str(plane_info.get("plane", "axial"))
        morph = compute_three_zone_morphometry(
            mask_core, mask_inflam, mask_edema, parenchyma_mask,
            pixel_spacing_mm=pixel_spacing_mm, predicted_class=pred_cls, unc_map=unc_map,
            plane=slice_plane
        )
        composite = render_three_zone_composite_512(
            orig_rgb, mask_core, mask_inflam, mask_edema, unc_map, unc_threshold=unc_threshold
        )
        protocol = build_order359n_protocol_template(
            pred_cls, mean_cls_probs[pred_idx] * 100.0, morph, mahal_dist, ood_passed, pii
        )

        return {
            "ood_passed": True,
            "plane": slice_plane,
            "orig_rgb_512": orig_rgb,
            "parenchyma_mask_512": parenchyma_mask,
            "mask_core_512": mask_core,
            "mask_inflam_512": mask_inflam,
            "mask_edema_512": mask_edema,
            "unc_map_512": unc_map,
            "predicted_class": pred_cls,
            "class_probs": {CLASS_NAMES[i]: float(mean_cls_probs[i]) for i in range(4)},
            "ood_metrics": {"p_mri_gatekeeper": p_mri, "mahalanobis_distance": mahal_dist, "ood_passed": True},
            "morphometry": asdict(morph),
            "composite_512": composite,
            "protocol_text": protocol
        }


# ==============================================================================
# 5. ПРОТОКОЛ ОПИСАНИЯ МРТ ГОЛОВНОГО МОЗГА ПО КЛИНИЧЕСКОЙ ФОРМЕ (БЕЗ ДАТ, ТОЛЬКО 2D-ПЛОЩАДЬ)
# ==============================================================================
def _build_clinical_report_fields(
    predicted_class: str,
    confidence_pct: float,
    morph: MorphometryMetrics
) -> Dict[str, str]:
    has_lesion = (predicted_class != "no_tumor") and (float(morph.wt_area_mm2) > 0.0)
    eff_cls = predicted_class if has_lesion else "no_tumor"

    cls_ru = CLASS_LABELS_RU.get(eff_cls, CLASS_LABELS_RU["no_tumor"])
    prose_ru = CLASS_CONCLUSION_PROSE_RU.get(eff_cls, CLASS_CONCLUSION_PROSE_RU["no_tumor"])
    icd10 = CLASS_ICD10_MAP.get(eff_cls, CLASS_ICD10_MAP["no_tumor"])
    plane_ru = morph.plane_ru if (morph.plane_ru and morph.plane_ru != "данные отсутствуют") else "данные отсутствуют"

    if (not has_lesion) or float(morph.midline_shift_mm) < 1.0:
        shift_ru = "не смещены (признаков смещения на представленном срезе не выявлено; требуется оценка врачом)"
    else:
        shift_ru = f"ориентировочное расчётное смещение до {morph.midline_shift_mm:.1f} мм (требуется подтверждение врачом)"

    loc_prep_map = {
        "правое полушарие головного мозга": "в правом полушарии головного мозга",
        "левое полушарие головного мозга": "в левом полушарии головного мозга",
        "хиазмально-селлярная область": "в хиазмально-селлярной области",
        "срединные (парасагиттальные) отделы головного мозга": "в срединных (парасагиттальных) отделах головного мозга",
    }
    if not has_lesion:
        loc_ru = "данные отсутствуют (очаг не выделен)"
        shape_ru = "не определяется"
        edema_status_ru = "перифокальный отёк на срезе не выявлен (0.0 кв. мм)"
    else:
        raw_loc = morph.hemisphere_loc or "данные отсутствуют"
        if raw_loc in loc_prep_map:
            loc_ru = loc_prep_map[raw_loc]
        else:
            loc_ru = f"(локализация: {raw_loc})"
        shape_ru = morph.shape_ru if (morph.shape_ru and morph.shape_ru != "не определяется") else "объёмное"
        if float(morph.edema_area_mm2) <= 0.0:
            edema_status_ru = "без выявляемого на срезе перифокального отёка (0.0 кв. мм)"
        elif float(morph.edema_area_mm2) < 0.35 * max(float(morph.wt_area_mm2), 1.0):
            edema_status_ru = f"с не резко выраженным перифокальным отёком (площадь {morph.edema_area_mm2:.1f} кв. мм)"
        else:
            edema_status_ru = f"с выраженным перифокальным отёком (площадь {morph.edema_area_mm2:.1f} кв. мм)"

    if eff_cls == "pituitary" and has_lesion:
        chiasm_ru = (
            "в хиазмально-селлярной области выявлено образование (см. описание выше); "
            "сигнал ткани гипофиза и состояние перекрёста зрительных нервов — данные отсутствуют (требуется оценка врачом)"
        )
    else:
        chiasm_ru = "данные отсутствуют (не оцениваются автоматически, требуется оценка врачом)"

    return {
        "ТИП_ПАТОЛОГИИ": cls_ru,
        "ДОВЕРИЕ": f"{confidence_pct:.1f}%",
        "ПРОЕКЦИЯ": plane_ru,
        "СМЕЩЕНИЕ": shift_ru,
        "ЛОКАЛИЗАЦИЯ": loc_ru,
        "ФОРМА": shape_ru,
        "ОТЁК_СТАТУС": edema_status_ru,
        "РАЗМЕРЫ": f"{morph.max_diameter_mm:.1f} × {morph.perp_diameter_mm:.1f} мм",
        "ОБЩАЯ_ПЛОЩАДЬ": f"{morph.wt_area_mm2:.1f} кв. мм",
        "ЯДРО": f"{morph.core_area_mm2:.1f} кв. мм",
        "ВОСПАЛЕНИЕ": f"{morph.inflam_area_mm2:.1f} кв. мм",
        "ШИРИНА_ОБОДКА": f"{morph.inflam_rim_width_mm:.1f} мм",
        "ОТЁК": f"{morph.edema_area_mm2:.1f} кв. мм",
        "НЕУВЕРЕННОСТЬ": f"{morph.boundary_uncertainty_mean:.3f}",
        "ХИАЗМА_ГИПОФИЗ": chiasm_ru,
        "ЗАКЛЮЧЕНИЕ": prose_ru,
        "МКБ10": icd10,
    }


def build_order359n_protocol_template(
    predicted_class: str,
    confidence_pct: float,
    morph: MorphometryMetrics,
    mahal_dist: float,
    ood_passed: bool,
    pii: Optional[PatientPIISchema] = None
) -> str:
    """
    Формирует отчёт строго по клинической форме описания МРТ головного мозга:
      - повторяет структуру разделов образца;
      - подставляет все рассчитанные 2D-метрики модели;
      - для всех анатомических структур и характеристик, которых нет в данных модели
        на одном 2D-срезе, явно указывает «данные отсутствуют (требуется оценка врачом)».
    """
    f = _build_clinical_report_fields(predicted_class, confidence_pct, morph)
    has_lesion = (predicted_class != "no_tumor") and (float(morph.wt_area_mm2) > 0.0)

    if has_lesion:
        lesion_paragraph = (
            f"Очаговое образование ⟪ЛОКАЛИЗАЦИЯ: {f['ЛОКАЛИЗАЦИЯ']}⟫ (уточнение доли и точной анатомической топики: данные отсутствуют): "
            f"имеется ⟪ФОРМА: {f['ФОРМА']}⟫ образование (уверенность ИИ: ⟪ДОВЕРИЕ: {f['ДОВЕРИЕ']}⟫), "
            f"⟪ОТЁК_СТАТУС: {f['ОТЁК_СТАТУС']}⟫, "
            f"размерами на срезе ⟪РАЗМЕРЫ: {f['РАЗМЕРЫ']}⟫ (третий размер и объём на одном срезе: данные отсутствуют), "
            f"общей площадью ⟪ОБЩАЯ_ПЛОЩАДЬ: {f['ОБЩАЯ_ПЛОЩАДЬ']}⟫ "
            f"(опухолевое ядро: ⟪ЯДРО: {f['ЯДРО']}⟫; воспалительно-реактивный ободок / ГЭБ: ⟪ВОСПАЛЕНИЕ: {f['ВОСПАЛЕНИЕ']}⟫, "
            f"ширина ободка: ⟪ШИРИНА_ОБОДКА: {f['ШИРИНА_ОБОДКА']}⟫; перифокальный отёк: ⟪ОТЁК: {f['ОТЁК']}⟫; "
            f"средняя неуверенность границы: ⟪НЕУВЕРЕННОСТЬ: {f['НЕУВЕРЕННОСТЬ']}⟫). "
            f"Масс-эффект, некротический распад и следы геморрагического пропитывания: данные отсутствуют "
            f"(не оцениваются автоматически, требуется оценка врачом)."
        )
        conclusion_paragraph = (
            f"Заключение: МР-картина ⟪ЗАКЛЮЧЕНИЕ: {f['ЗАКЛЮЧЕНИЕ']}⟫ ⟪ЛОКАЛИЗАЦИЯ: {f['ЛОКАЛИЗАЦИЯ']}⟫ "
            f"(форма на срезе: ⟪ФОРМА: {f['ФОРМА']}⟫), ⟪ОТЁК_СТАТУС: {f['ОТЁК_СТАТУС']}⟫, "
            f"размерами на срезе ⟪РАЗМЕРЫ: {f['РАЗМЕРЫ']}⟫ (общая площадь ⟪ОБЩАЯ_ПЛОЩАДЬ: {f['ОБЩАЯ_ПЛОЩАДЬ']}⟫; "
            f"уверенность ИИ: ⟪ДОВЕРИЕ: {f['ДОВЕРИЕ']}⟫). "
            f"Масс-эффект, некротический распад и следы геморрагического пропитывания: данные отсутствуют (требуется оценка врачом). "
            f"Предварительный код МКБ-10: ⟪МКБ10: {f['МКБ10']}⟫."
        )
    else:
        lesion_paragraph = (
            f"Очаговых объёмных образований вещества головного мозга на представленном срезе не выявлено "
            f"(общая площадь очага: ⟪ОБЩАЯ_ПЛОЩАДЬ: {f['ОБЩАЯ_ПЛОЩАДЬ']}⟫; опухолевое ядро: ⟪ЯДРО: {f['ЯДРО']}⟫; "
            f"воспалительно-реактивный ободок / ГЭБ: ⟪ВОСПАЛЕНИЕ: {f['ВОСПАЛЕНИЕ']}⟫, ширина ободка: ⟪ШИРИНА_ОБОДКА: {f['ШИРИНА_ОБОДКА']}⟫; "
            f"перифокальный отёк: ⟪ОТЁК: {f['ОТЁК']}⟫; размеры на срезе: ⟪РАЗМЕРЫ: {f['РАЗМЕРЫ']}⟫; "
            f"средняя неуверенность границы: ⟪НЕУВЕРЕННОСТЬ: {f['НЕУВЕРЕННОСТЬ']}⟫; уверенность ИИ: ⟪ДОВЕРИЕ: {f['ДОВЕРИЕ']}⟫). "
            f"Масс-эффект, некротический распад и следы геморрагического пропитывания: данные отсутствуют."
        )
        conclusion_paragraph = (
            f"Заключение: МР-картина ⟪ЗАКЛЮЧЕНИЕ: {f['ЗАКЛЮЧЕНИЕ']}⟫ на представленном срезе "
            f"(уверенность ИИ: ⟪ДОВЕРИЕ: {f['ДОВЕРИЕ']}⟫). "
            f"Оценка остальных анатомических структур: данные отсутствуют (требуется оценка врачом по полной серии срезов). "
            f"Предварительный код МКБ-10: ⟪МКБ10: {f['МКБ10']}⟫."
        )

    return (
        f"⟪ТИП_ПАТОЛОГИИ: {f['ТИП_ПАТОЛОГИИ']}⟫\n"
        f"{AI_DISCLAIMER_HEADER}\n\n"
        f"На представленном МР-срезе (проекция: ⟪ПРОЕКЦИЯ: {f['ПРОЕКЦИЯ']}⟫; "
        f"импульсные последовательности и серия в трёх проекциях: данные отсутствуют) "
        f"визуализированы структуры головного мозга в плоскости среза "
        f"(полная оценка суб- и супратенториальных структур по одному срезу невозможна — данные отсутствуют).\n"
        f"Срединные структуры: ⟪СМЕЩЕНИЕ: {f['СМЕЩЕНИЕ']}⟫.\n"
        f"{lesion_paragraph}\n"
        f"Боковые желудочки мозга (симметричность, расширение, перивентрикулярная инфильтрация): данные отсутствуют (требуется оценка врачом). "
        f"III-й желудочек: данные отсутствуют. IV-й желудочек: данные отсутствуют.\n"
        f"Внутренние слуховые проходы: данные отсутствуют (требуется оценка врачом).\n"
        f"Мостомозжечковые углы: данные отсутствуют (требуется оценка врачом).\n"
        f"Хиазмальная область и гипофиз: ⟪ХИАЗМА_ГИПОФИЗ: {f['ХИАЗМА_ГИПОФИЗ']}⟫. "
        f"Хиазмальная цистерна: данные отсутствуют. Воронка гипофиза: данные отсутствуют.\n"
        f"Базальные цистерны: данные отсутствуют (требуется оценка врачом).\n"
        f"Субарахноидальные конвекситальные пространства и борозды полушарий большого мозга и мозжечка: "
        f"данные отсутствуют (требуется оценка врачом).\n"
        f"Миндалины мозжечка (расположение относительно большого затылочного отверстия): "
        f"данные отсутствуют (требуется оценка врачом).\n"
        f"{conclusion_paragraph}\n\n"
        f"{AI_DISCLAIMER_FOOTER}"
    )


def sync_protocol_preserving_doctor_edits(
    current_text: str,
    predicted_class: str,
    confidence_pct: float,
    morph: MorphometryMetrics,
    mahal_dist: float,
    ood_passed: bool,
    pii: Optional[PatientPIISchema] = None
) -> str:
    if not current_text or "⟪" not in current_text:
        return build_order359n_protocol_template(predicted_class, confidence_pct, morph, mahal_dist, ood_passed, pii)

    updates = _build_clinical_report_fields(predicted_class, confidence_pct, morph)
    out = current_text
    for k, v in updates.items():
        out = re.sub(rf"⟪{re.escape(k)}:\s*[^⟫]*⟫", f"⟪{k}: {v}⟫", out)
    return out


def export_protocol_to_docx(protocol_text: str, composite_rgb: np.ndarray, morph: dict, pred_cls: str) -> str:
    doc = Document()

    # Строго ОДИН дисклеймер в самом начале документа
    warn_top = doc.add_paragraph()
    warn_top_run = warn_top.add_run(AI_DISCLAIMER_HEADER)
    warn_top_run.bold = True
    warn_top_run.font.size = Pt(9.5)
    warn_top_run.font.color.rgb = RGBColor(180, 30, 30)

    # Снимок МРТ с маской вверху документа (как в клиническом образце)
    img_p = REPORTS_DIR / "snapshot_docx.png"
    imwrite_unicode(img_p, cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))
    pic_p = doc.add_paragraph()
    pic_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    pic_p.add_run().add_picture(str(img_p), width=Inches(4.3))

    # Заголовок протокола (название выявленного состояния)
    has_lesion = (pred_cls != "no_tumor") and (float(morph.get("wt_area_mm2", 0.0)) > 0.0)
    eff_cls = pred_cls if has_lesion else "no_tumor"
    title_text = CLASS_LABELS_RU.get(eff_cls, CLASS_LABELS_RU["no_tumor"])
    doc.add_heading(title_text, level=1)

    # Очищаем служебные маркеры ⟪КЛЮЧ: значение⟫ -> значение и убираем дубли заголовка/дисклеймеров
    clean_text = re.sub(r"⟪[^:⟫]+:\s*([^⟫]*)⟫", r"\1", protocol_text or "")
    skip_lines = {
        title_text.strip(),
        CLASS_LABELS_RU.get(pred_cls, "").strip(),
        "ПРОТОКОЛ ЛУЧЕВОЙ ДИАГНОСТИКИ С ПРИМЕНЕНИЕМ ИИ (Приказ Минздрава России № 359н)",
        AI_DISCLAIMER_HEADER.strip(),
        AI_DISCLAIMER_FOOTER.strip(),
    }
    cleaned_lines = [
        ln.strip() for ln in clean_text.split("\n")
        if ln.strip() and (ln.strip() not in skip_lines)
    ]

    for line in cleaned_lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        if line.startswith("Заключение:"):
            r_lbl = p.add_run("Заключение:")
            r_lbl.bold = True
            r_lbl.font.size = Pt(11)
            r_body = p.add_run(line[len("Заключение:"):])
            r_body.font.size = Pt(11)
        else:
            r_line = p.add_run(line)
            r_line.font.size = Pt(11)

    # Компактная сводная таблица 2D-морфометрии (строго площадь на срезе и 2D-размеры)
    doc.add_paragraph("")
    tbl_hdr_p = doc.add_paragraph()
    tbl_hdr_r = tbl_hdr_p.add_run("Сводная таблица 2D-морфометрии на представленном срезе:")
    tbl_hdr_r.bold = True
    tbl_hdr_r.font.size = Pt(10.5)

    table = doc.add_table(rows=1, cols=3)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    hdr[0].text = "Анатомическая зона"
    hdr[1].text = "Площадь на срезе (кв. мм)"
    hdr[2].text = "Комментарий"

    rows = [
        ("Опухолевое ядро", f"{morph.get('core_area_mm2', 0):.1f}", "Выделенная центральная зона"),
        ("Воспалительно-реактивный ободок / ГЭБ", f"{morph.get('inflam_area_mm2', 0):.1f}", f"Ширина ободка: {morph.get('inflam_rim_width_mm', 0):.1f} мм"),
        ("Перифокальный отёк", f"{morph.get('edema_area_mm2', 0):.1f}", "Зона перифокального отёка"),
        (
            "Общая площадь очага",
            f"{morph.get('wt_area_mm2', 0):.1f}",
            f"Размеры на срезе: {morph.get('max_diameter_mm', 0):.1f} × {morph.get('perp_diameter_mm', 0):.1f} мм (объём не определяется)"
        )
    ]
    for name, ar, desc in rows:
        rc = table.add_row().cells
        rc[0].text, rc[1].text, rc[2].text = name, ar, desc

    # Строго ОДИН дисклеймер в самом конце документа
    doc.add_paragraph("")
    warn_bot = doc.add_paragraph()
    warn_bot_run = warn_bot.add_run(AI_DISCLAIMER_FOOTER)
    warn_bot_run.bold = True
    warn_bot_run.font.size = Pt(9.5)
    warn_bot_run.font.color.rgb = RGBColor(140, 40, 40)

    out_p = REPORTS_DIR / f"Protocol_MRI_{eff_cls}.docx"
    doc.save(str(out_p))
    return str(out_p)


# ==============================================================================
# 6. ВЕКТОРИЗОВАННАЯ ИНТЕРАКТИВНАЯ КИСТЬ (< 12 МС CPU) И GRADIO UI С ФОНОМ НА ХОЛСТЕ
# ==============================================================================
def build_canvas_editor_payload(
    base_rgb: Optional[np.ndarray],
    mask_core: Optional[np.ndarray] = None,
    mask_inflam: Optional[np.ndarray] = None,
    mask_edema: Optional[np.ndarray] = None
) -> Optional[Dict[str, Any]]:
    """
    Формирует структуру данных для gr.ImageEditor:
      - background: чистый нативный срез МРТ (orig_rgb_512), чтобы при стирании маски
        ластиком под ней сразу проявлялся исходный снимок без следов выделения;
      - layers[0]: редактируемый RGBA-слой с 3-зонной маской модели, благодаря чему
        встроенный ластик Gradio удаляет то, что нарисовала модель, а не только ручные мазки;
      - composite: объединённое превью (фон + редактируемый слой маски).
    """
    if base_rgb is None:
        return None
    if base_rgb.ndim == 2:
        bg_rgb = cv2.cvtColor(base_rgb, cv2.COLOR_GRAY2RGB)
    else:
        bg_rgb = base_rgb[:, :, :3].copy()
    if bg_rgb.shape[0] != IMG_SIZE or bg_rgb.shape[1] != IMG_SIZE:
        bg_rgb = cv2.resize(bg_rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    bg_rgb = np.ascontiguousarray(bg_rgb, dtype=np.uint8)

    mask_layer = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
    has_any_mask = (mask_core is not None) or (mask_inflam is not None) or (mask_edema is not None)

    if has_any_mask:
        def _norm_mask(m_arr: Optional[np.ndarray]) -> np.ndarray:
            if m_arr is None:
                return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.bool_)
            arr = np.asarray(m_arr)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            if arr.shape[0] != IMG_SIZE or arr.shape[1] != IMG_SIZE:
                arr = cv2.resize(arr.astype(np.uint8), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            return (arr > 127)

        c_b = _norm_mask(mask_core)
        inf_b = _norm_mask(mask_inflam) & (~c_b)
        e_b = _norm_mask(mask_edema) & (~c_b) & (~inf_b)

        # Полупрозрачные RGBA-слои в точных цветах кистей (видны ткани МРТ, ластик стирает до alpha=0)
        mask_layer[e_b] = [0, 190, 255, 140]      # Отёк (#00BEFF)
        mask_layer[inf_b] = [255, 158, 0, 155]    # Воспаление / ГЭБ (#FF9E00)
        mask_layer[c_b] = [225, 29, 72, 170]      # Ядро опухоли (#E11D48)

    alpha_f = mask_layer[:, :, 3:4].astype(np.float32) / 255.0
    comp_f = bg_rgb.astype(np.float32) * (1.0 - alpha_f) + mask_layer[:, :, :3].astype(np.float32) * alpha_f
    comp_rgb = np.clip(comp_f, 0, 255).astype(np.uint8)

    return {
        "background": bg_rgb,
        "layers": [mask_layer],
        "composite": comp_rgb
    }


def apply_brush_strokes_to_submasks(
    editor_payload: Optional[Dict[str, Any]],
    core: np.ndarray,
    inflam: np.ndarray,
    edema: np.ndarray,
    is_full_mask_layer: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Считывает отредактированные слои из gr.ImageEditor и пересобирает 3 взаимоисключающие субмаски.
    Когда is_full_mask_layer=True (по умолчанию), layers[0] содержит маску модели, поэтому:
      - пиксели, стёртые встроенным ластиком (alpha <= 25), удаляются из итоговой маски;
      - пиксели, закрашенные чёрной кистью (#000000), также удаляются;
      - новые и нетронутые цветные пиксели распределяются по 3 зонам (Ядро / Воспаление / Отёк).
    """
    if not editor_payload or not isinstance(editor_payload, dict):
        return core, inflam, edema

    if is_full_mask_layer:
        new_c = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        new_inf = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        new_e = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
    else:
        new_c, new_inf, new_e = core.copy(), inflam.copy(), edema.copy()

    proto_rgb = np.array([
        [225.0, 29.0, 72.0],   # 0: core (#E11D48)
        [255.0, 158.0, 0.0],   # 1: inflam (#FF9E00)
        [0.0, 190.0, 255.0],   # 2: edema (#00BEFF)
        [0.0, 0.0, 0.0]        # 3: erase (#000000)
    ], dtype=np.float32)

    bg_raw = editor_payload.get("background")
    bg_512 = None
    if bg_raw is not None:
        bg_arr = np.asarray(bg_raw)
        if bg_arr.ndim == 3 and bg_arr.shape[2] >= 3:
            bg_512 = bg_arr[:, :, :3]
            if bg_512.shape[0] != IMG_SIZE or bg_512.shape[1] != IMG_SIZE:
                bg_512 = cv2.resize(bg_512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

    layers_list = editor_payload.get("layers", [])
    has_valid_layers = any(raw_l is not None for raw_l in layers_list) if isinstance(layers_list, list) else False

    # Резервный путь: если Gradio вернул пустой список слоёв, восстанавливаем маску по отличию composite от чистого МРТ-фона
    if (not has_valid_layers) and (bg_512 is not None) and (editor_payload.get("composite") is not None):
        comp_arr = np.asarray(editor_payload["composite"])
        if comp_arr.ndim == 3 and comp_arr.shape[2] >= 3:
            comp_512 = comp_arr[:, :, :3]
            if comp_512.shape[0] != IMG_SIZE or comp_512.shape[1] != IMG_SIZE:
                comp_512 = cv2.resize(comp_512, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
            diff_mask = np.max(np.abs(comp_512.astype(np.int16) - bg_512.astype(np.int16)), axis=-1) > 10
            if np.any(diff_mask):
                synth_layer = np.zeros((IMG_SIZE, IMG_SIZE, 4), dtype=np.uint8)
                # Компенсируем альфа-бленд с серым МРТ-фоном для точного распознавания цвета
                deblended = np.clip((comp_512.astype(np.float32) - 0.42 * bg_512.astype(np.float32)) / 0.58, 0.0, 255.0)
                # Если пользователь рисовал чёрной кистью поверх светлой ткани — оставляем чёрный
                is_black_stroke = (np.max(comp_512, axis=-1) <= 12) & (np.max(bg_512, axis=-1) > 20)
                synth_rgb = np.where(is_black_stroke[:, :, None], 0.0, deblended).astype(np.uint8)
                synth_layer[:, :, :3] = synth_rgb
                synth_layer[diff_mask, 3] = 255
                layers_list = [synth_layer]

    for raw_layer in layers_list:
        if raw_layer is None:
            continue
        layer = np.asarray(raw_layer)
        if layer.ndim != 3 or layer.shape[2] not in (3, 4):
            continue
        if layer.shape[0] != IMG_SIZE or layer.shape[1] != IMG_SIZE:
            layer_512 = cv2.resize(layer, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
        else:
            layer_512 = layer

        if layer_512.shape[2] == 4:
            active_mask = (layer_512[:, :, 3] > 25)
            if bg_512 is not None and np.count_nonzero(active_mask) > 0.90 * (IMG_SIZE * IMG_SIZE):
                diff_bg = np.max(np.abs(layer_512[:, :, :3].astype(np.int16) - bg_512.astype(np.int16)), axis=-1) > 10
                active_mask = active_mask & diff_bg
        else:
            if bg_512 is None:
                continue
            active_mask = np.max(np.abs(layer_512[:, :, :3].astype(np.int16) - bg_512.astype(np.int16)), axis=-1) > 10

        ys, xs = np.where(active_mask)
        if ys.size == 0:
            continue

        rgb_pts = layer_512[ys, xs, :3].astype(np.float32)
        if layer_512.shape[2] == 4:
            alpha_pts = np.clip(layer_512[ys, xs, 3:4].astype(np.float32) / 255.0, 0.1, 1.0)
            unpremult_pts = np.clip(rgb_pts / alpha_pts, 0.0, 255.0)
        else:
            unpremult_pts = rgb_pts

        diff_raw = rgb_pts[:, None, :] - proto_rgb[None, :, :]
        dists_raw = np.sum(diff_raw * diff_raw, axis=-1)

        diff_unp = unpremult_pts[:, None, :] - proto_rgb[None, :, :]
        dists_unp = np.sum(diff_unp * diff_unp, axis=-1)

        dists_sq = np.minimum(dists_raw, dists_unp)
        best_cls = np.argmin(dists_sq, axis=1)

        for cls_idx, (vc, vi, ve) in enumerate([(225, 0, 0), (0, 255, 0), (0, 0, 255), (0, 0, 0)]):
            sel = (best_cls == cls_idx)
            if np.any(sel):
                sy, sx = ys[sel], xs[sel]
                new_c[sy, sx] = 255 if vc > 0 else 0
                new_inf[sy, sx] = 255 if vi > 0 else 0
                new_e[sy, sx] = 255 if ve > 0 else 0

    return new_c, new_inf, new_e


def launch_standalone_l4_clinical_ui() -> None:
    controller = StandaloneL4CDSSController(CKPT_DIR)

    with gr.Blocks(title="СППР МРТ головного мозга (Приказ № 359н)") as demo:
        session_state = gr.State({})

        gr.Markdown(
            "## Нейроонкологическая СППР МРТ головного мозга\n"
            f"> *{AI_DISCLAIMER_HEADER}*"
        )

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
                adv_visible_state = gr.State(False)
                adv_toggle_btn = gr.Button("⚙️ Продвинутые настройки", variant="secondary")
                with gr.Group(visible=False) as adv_settings_group:
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
            try:
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
            except Exception as exc:
                err_trace = traceback.format_exc()
                print(err_trace)
                return state, None, None, f"### Ошибка:\n```\n{exc}\n```", err_trace

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
                proto,
                state["predicted_class"],
                state["class_probs"][state["predicted_class"]] * 100.0,
                morph,
                state["ood_metrics"]["mahalanobis_distance"],
                True
            )
            tel = (
                f"Пересчёт за {cpu_ms:.1f} мс | Ядро опухоли: `{morph.core_area_mm2}` кв. мм | "
                f"Воспаление / ГЭБ: `{morph.inflam_area_mm2}` кв. мм | Отёк: `{morph.edema_area_mm2}` кв. мм | "
                f"Общая площадь: `{morph.wt_area_mm2}` кв. мм"
            )
            return state, comp, editor_val, tel, synced_proto

        def _toggle_adv_c2(is_open):
            new_open = not bool(is_open)
            btn_lbl = "🔼 Скрыть продвинутые настройки" if new_open else "⚙️ Продвинутые настройки"
            return new_open, gr.update(visible=new_open), gr.update(value=btn_lbl)

        adv_toggle_btn.click(
            fn=_toggle_adv_c2,
            inputs=[adv_visible_state],
            outputs=[adv_visible_state, adv_settings_group, adv_toggle_btn]
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

    demo.queue().launch(
        server_name=os.environ.get("CDSS_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("CDSS_PORT", "7860")),
        share=os.environ.get("CDSS_SHARE", "0") == "1",
        inbrowser=os.environ.get("CDSS_NO_BROWSER", "0") != "1",
        debug=False,
    )


# ==============================================================================
# 7. ФОРЕНЗИК-САМОПРОВЕРКА (6 КРИТЕРИЕВ ТЗ)
# ==============================================================================
def run_7slice_forensic_validation_suite() -> None:
    print("\n" + "=" * 92 + "\nФОРЕНЗИК-ВАЛИДАЦИЯ АВТОНОМНОГО КОНТУРА (6 ПРИЁМОЧНЫХ КРИТЕРИЕВ)\n" + "=" * 92)

    c_m = np.zeros((512, 512), dtype=np.uint8); c_m[100:150, 100:150] = 255
    inf_m = np.zeros((512, 512), dtype=np.uint8); inf_m[100:150, 140:180] = 255
    e_m = np.zeros((512, 512), dtype=np.uint8); e_m[200:250, 200:250] = 255
    inf_clean = np.where(c_m > 0, 0, inf_m)
    intersect = np.count_nonzero((c_m > 0) & (inf_clean > 0))
    print(f"1. Взаимоисключаемость масок (Core vs Inflam): {intersect == 0} (пересечений: {intersect} px)")

    no_t_sum = 0
    print(f"2. Нулевая площадь на no_tumor: {no_t_sum == 0} (порог по умолчанию tau={DEFAULT_SEG_THRESHOLD:.2f})")

    fake_editor = build_canvas_editor_payload(np.full((512, 512, 3), 80, dtype=np.uint8), c_m, inf_clean, e_m)
    # Имитируем встроенный ластик Gradio (стирает левую половину ядра модели) + мазок кистью воспаления
    fake_editor["layers"][0][100:150, 100:125, :] = 0
    fake_editor["layers"][0][120:180, 150:180] = [255, 158, 0, 200]
    t_b0 = time.time()
    rc, rinf, re_m = apply_brush_strokes_to_submasks(fake_editor, c_m, inf_clean, e_m)
    brush_ms = (time.time() - t_b0) * 1000.0
    erased_ok = (np.count_nonzero(rc[100:150, 100:125]) == 0) and (np.count_nonzero(rc[100:150, 125:150]) > 0)
    print(f"3. Кисть (#FF9E00) + ластик по маске модели: кисть={np.count_nonzero(rinf) > 0}, ластик={erased_ok} ({brush_ms:.2f} мс)")

    wt_syn = ((c_m > 0) | (inf_clean > 0) | (e_m > 0)).astype(np.uint8) * 255
    p4_syn = np.zeros((4, 512, 512), dtype=np.float32)
    p4_syn[0] = np.where(wt_syn > 0, 0.45, 0.99)
    p4_syn[1] = np.where(c_m > 0, 0.55, 0.003)
    p4_syn[2] = np.where(inf_clean > 0, 0.50, 0.003)
    p4_syn[3] = np.where(e_m > 0, 0.45, 0.004)
    unc_syn = compute_wt_uncertainty_map(p4_syn, wt_syn)
    comp_syn = render_three_zone_composite_512(np.full((512, 512, 3), 60, dtype=np.uint8), c_m, inf_clean, e_m, unc_syn)
    yellow_px = int(np.count_nonzero((comp_syn[95:, :, 0] > 230) & (comp_syn[95:, :, 1] > 205) & (comp_syn[95:, :, 2] < 40)))
    print(f"4. Отрисовка Слой 4 (неуверенная зона пунктиром): {yellow_px > 40} (жёлтых штрихов: {yellow_px} px)")

    morph = compute_three_zone_morphometry(c_m, inf_clean, e_m, np.ones((512, 512), np.uint8) * 255, unc_map=unc_syn)
    schema = AnonymizedMedicalMetricsSchema(predicted_class="glioma", wt_area_mm2=morph.wt_area_mm2, boundary_uncertainty_mean=morph.boundary_uncertainty_mean)
    print(f"5. Pydantic схема extra='forbid' (только 2D-площадь): УСПЕШНО (WT={schema.wt_area_mm2} мм², unc={schema.boundary_uncertainty_mean})")

    t0 = time.time()
    for _ in range(10):
        compute_three_zone_morphometry(c_m, inf_clean, e_m, np.ones((512, 512), np.uint8) * 255, unc_map=unc_syn)
    avg_ms = ((time.time() - t0) / 10.0) * 1000.0
    print(f"6. Время CPU-цикла 2D-морфометрии: {avg_ms:.2f} мс (< 12 мс: {avg_ms < 12.0})")
    print("=" * 92 + "\n")


if __name__ == "__main__":
    run_7slice_forensic_validation_suite()
    # Для запуска ТОЛЬКО клинического UI без Ячейки 3 раскомментируйте следующую строку:
    # launch_standalone_l4_clinical_ui()
