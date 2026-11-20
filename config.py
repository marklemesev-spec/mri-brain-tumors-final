# -*- coding: utf-8 -*-
"""Конфигурация публичных ссылок и параметров по умолчанию для СППР МРТ 512x512."""

import os

# 1. Веса модели (checkpoints_512)
PUBLIC_GDRIVE_FOLDER_URL = os.environ.get(
    "CDSS_CKPT_FOLDER_URL",
    "https://drive.google.com/drive/folders/13zJLOabi522wS8vWmBwwFUCn73-Cz8mz?usp=sharing",
)

PUBLIC_GDRIVE_FILES = {
    "multitask_512_best.pth": os.environ.get(
        "CDSS_MULTITASK_URL",
        "https://drive.google.com/file/d/1qPfcCdxF36FvBbwBBG1QEwbbh34vCQmh/view?usp=sharing",
    ),
    "mahalanobis_ood_512.npz": os.environ.get(
        "CDSS_MAHAL_URL",
        "https://drive.google.com/file/d/1U-AtCWAWtL9tB1XHIrifvP2w74N9Bpvz/view?usp=sharing",
    ),
    "gatekeeper_512_best.pth": os.environ.get(
        "CDSS_GATEKEEPER_URL",
        "https://drive.google.com/file/d/1M8QLeG2BE0_HyoaN8t43QyVX28kT_FWx/view?usp=drive_link",
    ),
}

# 2. Подборка из 20 МРТ для первой вкладки (демо-плашка 4x5)
PUBLIC_DEMO_20_GDRIVE_URL = os.environ.get(
    "CDSS_DEMO20_URL",
    "https://drive.google.com/drive/folders/1skQO2Pl2QSBK92DdsfIIbCWr_ZxPAI6a?usp=sharing",
)
PUBLIC_DEMO_20_GDRIVE_FILES = []

# Автосборка плашки при запуске UI (можно выключить флагом --no-demo20)
AUTO_RUN_DEMO_20_ON_STARTUP = os.environ.get("CDSS_AUTO_DEMO20", "1") == "1"