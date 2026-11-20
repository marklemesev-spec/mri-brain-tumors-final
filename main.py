# -*- coding: utf-8 -*-
"""
Главная точка входа Neuro-CDSS МРТ 512x512 (Windows / Linux / macOS).

Примеры запуска:
  python main.py                                    # Веб-интерфейс (3 вкладки) на http://127.0.0.1:7860
  python main.py --no-demo20                        # Запуск UI без автозагрузки 20 МРТ
  python main.py --port 8080 --share                # Другой порт + временная публичная ссылка Gradio
  python main.py --device cpu                       # Принудительный запуск на CPU (или cuda / mps)
  python main.py --image scan.jpg --out ./results   # Headless CLI-анализ одного МРТ (PNG + JSON + DOCX)
  python main.py --self-test                        # Форензик-самопроверка (6 критериев)
  python main.py --download-only                    # Только скачать веса модели с Google Drive
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Активируем MPS fallback и UTF-8 консоль до импорта PyTorch
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from compat import configure_console_utf8, imread_unicode, imwrite_unicode

configure_console_utf8()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Нейроонкологическая СППР МРТ головного мозга 512x512 (Windows / Linux / macOS)"
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Устройство вычислений (по умолчанию: auto — CUDA -> Apple MPS -> CPU)",
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="",
        help="Папка для весов, кэша и отчётов (по умолчанию: ./cdss_workspace)",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="",
        help="Путь к локальной папке с уже скачанными весами (multitask_512_best.pth и др.)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="IP-адрес веб-сервера Gradio (0.0.0.0 для доступа из локальной сети)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Порт веб-сервера Gradio (по умолчанию: 7860)",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Создать временную публичную ссылку *.gradio.live",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Не открывать браузер автоматически при запуске",
    )
    parser.add_argument(
        "--no-demo20",
        action="store_true",
        help="Отключить автозагрузку 20 МРТ с Google Диска при старте UI",
    )
    parser.add_argument(
        "--ui-simple",
        action="store_true",
        help="Запустить только упрощённый клинический UI (1 срез, без массового аудита)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Выполнить форензик-самопроверку контура (6 приёмочных критериев) и выйти",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Только скачать и проверить веса модели, не запуская интерфейс",
    )
    # Headless CLI-режим для одного среза
    parser.add_argument(
        "--image",
        type=str,
        default="",
        help="Путь к одному МРТ-срезу (.png/.jpg/.webp/.tif/.dcm) для анализа прямо в терминале",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="./cli_output",
        help="Папка для сохранения результатов CLI-анализа (PNG, JSON, TXT, DOCX)",
    )
    parser.add_argument(
        "--seg-threshold",
        type=float,
        default=0.80,
        help="Порог выделения опухоли (по умолчанию: 0.80)",
    )
    parser.add_argument(
        "--unc-threshold",
        type=float,
        default=0.42,
        help="Порог неуверенной зоны (по умолчанию: 0.42)",
    )
    parser.add_argument(
        "--pixel-spacing",
        type=float,
        default=0.9375,
        help="Масштаб пикселя в мм (по умолчанию: 0.9375; для DICOM берётся автоматически)",
    )
    return parser.parse_args()


def load_image_or_dicom(filepath: Path):
    """Загружает стандартное изображение (RGB) или DICOM (.dcm) с автонормализацией."""
    import cv2
    import numpy as np

    if filepath.suffix.lower() == ".dcm":
        import pydicom

        ds = pydicom.dcmread(str(filepath))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3:
            arr = arr[arr.shape[0] // 2]
        lo, hi = float(np.percentile(arr, 1.0)), float(np.percentile(arr, 99.5))
        norm = np.clip((arr - lo) / max(hi - lo, 1e-5) * 255.0, 0, 255).astype(np.uint8)
        rgb = cv2.cvtColor(norm, cv2.COLOR_GRAY2RGB)
        ps = getattr(ds, "PixelSpacing", None)
        px_sp = float(ps[0]) if ps else None
        return rgb, px_sp

    bgr = imread_unicode(filepath, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {filepath}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), None


def run_cli_single_image(args: argparse.Namespace) -> None:
    import cv2
    import shutil
    import cdss_core as core

    img_path = Path(args.image).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    img_rgb, dicom_spacing = load_image_or_dicom(img_path)
    px_spacing = dicom_spacing if dicom_spacing is not None else float(args.pixel_spacing)

    controller = core.StandaloneL4CDSSController(core.CKPT_DIR)
    if not controller.model_loaded:
        print("[ERROR] Основная модель multitask_512_best.pth не загружена.")
        sys.exit(1)

    res = controller.predict_and_quantify(
        img_rgb,
        seg_threshold=float(args.seg_threshold),
        unc_threshold=float(args.unc_threshold),
        pixel_spacing_mm=px_spacing,
    )

    stem = img_path.stem
    overlay_path = out_dir / f"{stem}_overlay.png"
    imwrite_unicode(overlay_path, cv2.cvtColor(res["composite_512"], cv2.COLOR_RGB2BGR))

    proto_txt_path = out_dir / f"{stem}_protocol.txt"
    proto_txt_path.write_text(res["protocol_text"], encoding="utf-8")

    metrics_payload = {
        "file": str(img_path),
        "ood_passed": res["ood_passed"],
        "predicted_class": res["predicted_class"],
        "predicted_class_ru": core.CLASS_LABELS_RU.get(res["predicted_class"], res["predicted_class"]),
        "class_probs": res["class_probs"],
        "ood_metrics": res["ood_metrics"],
        "morphometry": res["morphometry"],
    }
    json_path = out_dir / f"{stem}_metrics.json"
    json_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    docx_dest = None
    if res["ood_passed"]:
        docx_src = core.export_protocol_to_docx(
            res["protocol_text"],
            res["composite_512"],
            res["morphometry"],
            res["predicted_class"],
        )
        docx_dest = out_dir / f"{stem}_protocol.docx"
        shutil.copy2(docx_src, docx_dest)

    print("\n" + "=" * 78)
    print(f"Файл:   {img_path.name}")
    print(f"Класс:  {metrics_payload['predicted_class_ru']}")
    if res["ood_passed"]:
        m = res["morphometry"]
        print(
            f"Площадь очага: {m['wt_area_mm2']} кв. мм "
            f"(Ядро: {m['core_area_mm2']} | Воспаление/ГЭБ: {m['inflam_area_mm2']} | Отёк: {m['edema_area_mm2']})"
        )
        print(f"Размеры:       {m['max_diameter_mm']} x {m['perp_diameter_mm']} мм")
    print(f"Разметка PNG:  {overlay_path}")
    print(f"Метрики JSON:  {json_path}")
    print(f"Протокол TXT:  {proto_txt_path}")
    if docx_dest:
        print(f"Протокол DOCX: {docx_dest}")
    print("=" * 78 + "\n")


def main() -> None:
    args = parse_args()

    # Передаём CLI-параметры в модули до их импорта
    os.environ["CDSS_DEVICE"] = args.device
    if args.workspace:
        os.environ["CDSS_WORKSPACE"] = args.workspace
    if args.ckpt_dir:
        os.environ["CDSS_CKPT_DIR"] = args.ckpt_dir
    os.environ["CDSS_HOST"] = args.host
    os.environ["CDSS_PORT"] = str(args.port)
    os.environ["CDSS_SHARE"] = "1" if args.share else "0"
    os.environ["CDSS_NO_BROWSER"] = "1" if args.no_browser else "0"
    if args.no_demo20:
        os.environ["CDSS_AUTO_DEMO20"] = "0"
    if args.self_test:
        os.environ["CDSS_SKIP_DOWNLOAD"] = "1"

    if args.self_test:
        import cdss_core as core
        core.run_7slice_forensic_validation_suite()
        return

    if args.download_only:
        import cdss_core as core
        print(f"[DONE] Папка с весами: {core.CKPT_DIR}")
        return

    if args.image:
        run_cli_single_image(args)
        return

    if args.ui_simple:
        import cdss_core as core
        core.launch_standalone_l4_clinical_ui()
    else:
        import cdss_ui as ui
        ui.launch_interactive_cdss_with_batch_auditor_v8()


if __name__ == "__main__":
    main()