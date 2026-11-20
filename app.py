# -*- coding: utf-8 -*-
"""Точка входа для Hugging Face Spaces (Gradio)."""

import os

# Настройки сервера для Hugging Face Spaces
os.environ["CDSS_HOST"] = "0.0.0.0"
os.environ["CDSS_PORT"] = os.environ.get("PORT", "7860")
os.environ["CDSS_SHARE"] = "0"
os.environ["CDSS_NO_BROWSER"] = "1"
os.environ["CDSS_AUTO_DEMO20"] = "0"
os.environ["CDSS_DEVICE"] = "auto"

import cdss_ui as ui

if __name__ == "__main__":
    ui.launch_interactive_cdss_with_batch_auditor_v8()