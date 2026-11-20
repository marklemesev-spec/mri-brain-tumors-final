# -*- coding: utf-8 -*-
"""Кроссплатформенные утилиты совместимости для Windows, Linux и macOS."""

import os
import sys
import contextlib
from pathlib import Path
from typing import Tuple, Optional

# Включаем fallback на CPU для неподдерживаемых операций Apple Silicon MPS
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import numpy as np
import torch


def configure_console_utf8() -> None:
    """Предотвращает UnicodeEncodeError при выводе кириллицы и эмодзи в терминале Windows."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def get_workspace_dir() -> Path:
    """Возвращает рабочую директорию (чекпоинты, отчёты, кэш демо-снимков)."""
    env_ws = os.environ.get("CDSS_WORKSPACE", "").strip()
    if env_ws:
        base = Path(env_ws).expanduser().resolve()
    else:
        base = (Path(__file__).resolve().parent / "cdss_workspace").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def select_compute_device(preferred: Optional[str] = None) -> Tuple[torch.device, torch.dtype]:
    """
    Выбирает оптимальное устройство (cuda / mps / cpu) и тип AMP.
    Учитывает флаг --device / переменную окружения CDSS_DEVICE.
    """
    req = (preferred or os.environ.get("CDSS_DEVICE", "auto")).strip().lower()

    if req == "cuda":
        if torch.cuda.is_available():
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            return torch.device("cuda"), dtype
        print("[DEVICE-WARN] Запрошен CUDA, но GPU NVIDIA не найден. Переключение на CPU.")
        return torch.device("cpu"), torch.float32

    if req == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), torch.float32
        print("[DEVICE-WARN] Запрошен MPS, но Apple Silicon GPU недоступен. Переключение на CPU.")
        return torch.device("cpu"), torch.float32

    if req == "cpu":
        return torch.device("cpu"), torch.float32

    # Автовыбор: CUDA -> MPS -> CPU
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def amp_autocast(device: torch.device, amp_dtype: torch.dtype):
    """Безопасный autocast: активен только на CUDA с bfloat16, на CPU/MPS не вызывает ошибок."""
    if device.type == "cuda" and amp_dtype == torch.bfloat16:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True)
    return contextlib.nullcontext()


def imread_unicode(filepath: str | Path, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    """Безопасное чтение изображений OpenCV с поддержкой кириллицы в путях Windows."""
    try:
        p = Path(filepath)
        if not p.is_file():
            return None
        data = np.fromfile(str(p), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, flags)
    except Exception:
        return None


def imwrite_unicode(filepath: str | Path, image: np.ndarray) -> bool:
    """Безопасное сохранение изображений OpenCV с поддержкой кириллицы в путях Windows."""
    try:
        p = Path(filepath)
        p.parent.mkdir(parents=True, exist_ok=True)
        ext = p.suffix if p.suffix else ".png"
        ok, buf = cv2.imencode(ext, image)
        if not ok:
            return False
        buf.tofile(str(p))
        return True
    except Exception:
        return False


def get_windows_font_candidates() -> list[str]:
    """Возвращает пути к кириллическим шрифтам Windows."""
    windir = os.environ.get("WINDIR", r"C:\Windows")
    fonts_dir = Path(windir) / "Fonts"
    return [
        str(fonts_dir / "arialbd.ttf"),
        str(fonts_dir / "arial.ttf"),
        str(fonts_dir / "segoeuib.ttf"),
        str(fonts_dir / "segoeui.ttf"),
        str(fonts_dir / "tahomabd.ttf"),
    ]
def load_image_from_url(url: str) -> Tuple[Optional[np.ndarray], str]:
    """Скачивает МРТ-снимок по прямой ссылке или ссылке Google Диска и возвращает RGB-массив."""
    import tempfile
    import urllib.request

    url_clean = (url or "").strip()
    if not url_clean:
        return None, "Вставьте ссылку на изображение или файл Google Диска."
    if not url_clean.startswith(("http://", "https://")):
        return None, "Ссылка должна начинаться с http:// или https://"

    tmp_dir = Path(tempfile.mkdtemp(prefix="cdss_url_"))
    try:
        downloaded_path: Optional[Path] = None
        if "drive.google.com" in url_clean or "docs.google.com" in url_clean:
            import gdown
            out_prefix = str(tmp_dir) + os.sep
            res_p = gdown.download(url=url_clean, output=out_prefix, quiet=True, fuzzy=True)
            if res_p and Path(res_p).is_file():
                downloaded_path = Path(res_p)
        else:
            req = urllib.request.Request(url_clean, headers={"User-Agent": "Mozilla/5.0"})
            target_p = tmp_dir / "downloaded_slice"
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = resp.read()
            target_p.write_bytes(data)
            downloaded_path = target_p

        if downloaded_path is None or not downloaded_path.is_file():
            return None, "Не удалось скачать файл по ссылке. Проверьте, что доступ к файлу открыт."

        # Проверка на DICOM (.dcm или сигнатура DICM)
        raw_head = downloaded_path.read_bytes()[:132]
        is_dicom = downloaded_path.suffix.lower() == ".dcm" or raw_head[128:132] == b"DICM"
        if is_dicom:
            import pydicom
            ds = pydicom.dcmread(str(downloaded_path))
            arr = ds.pixel_array.astype(np.float32)
            if arr.ndim == 3:
                arr = arr[arr.shape[0] // 2]
            lo, hi = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.5))
            norm = np.clip((arr - lo) / max(hi - lo, 1e-5) * 255.0, 0, 255).astype(np.uint8)
            return cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB), "DICOM-снимок загружен по ссылке."

        bgr = imread_unicode(downloaded_path, cv2.IMREAD_COLOR)
        if bgr is None:
            return None, "Скачанный файл не является изображением МРТ (проверьте, что ссылка ведёт на сам файл, а не на веб-страницу)."
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "Снимок успешно загружен по ссылке."
    except Exception as exc:
        return None, f"Ошибка загрузки по ссылке: {exc}"
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


def make_copyable_textbox(gr_module, **kwargs):
    """Создаёт gr.Textbox с кнопкой копирования, совместимый с Gradio 4 и 5."""
    import inspect
    params = inspect.signature(gr_module.Textbox.__init__).parameters
    if "show_copy_button" in params:
        kwargs["show_copy_button"] = True
    return gr_module.Textbox(**kwargs)
