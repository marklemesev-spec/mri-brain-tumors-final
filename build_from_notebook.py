# -*- coding: utf-8 -*-
"""
Конвертер исходного Google Colab блокнота (.ipynb) в кроссплатформенные модули
cdss_core.py и cdss_ui.py для запуска из терминала на Windows, Linux и macOS.

Использование:
    python build_from_notebook.py path/to/your_notebook.ipynb
"""

import json
import py_compile
import re
import sys
from pathlib import Path


CORE_HEADER = '''# -*- coding: utf-8 -*-
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
'''

UI_HEADER = '''# -*- coding: utf-8 -*-
"""Автоматически сгенерировано из Ячейки 3 (кроссплатформенный UI и пакетный аудит)."""

import os
import tempfile
from pathlib import Path
from compat import (
    configure_console_utf8,
    amp_autocast,
    imread_unicode,
    load_image_from_url,
    make_copyable_textbox,
)
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
'''

NEW_SETUP_ENV_FN = '''def setup_standalone_l4_environment() -> Tuple[Path, Path]:
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
    return local_base, active_ckpt_dir'''


def patch_cell2(src: str) -> str:
    # 1. Заменяем setup_standalone_l4_environment (убираем runtime pip install и /content)
    pattern_setup = re.compile(
        r"def setup_standalone_l4_environment\(\) -> Tuple\[Path, Path\]:.*?return local_base, active_ckpt_dir",
        flags=re.DOTALL,
    )
    src, n_sub = pattern_setup.subn(NEW_SETUP_ENV_FN, src, count=1)
    if n_sub != 1:
        raise RuntimeError("Не удалось найти setup_standalone_l4_environment() в Ячейке 2.")

    # 2. Кроссплатформенный выбор устройства CUDA / MPS / CPU
    old_dev_block = (
        "torch.backends.cudnn.enabled = False\n"
        "torch.backends.cudnn.benchmark = False\n"
        "torch.set_float32_matmul_precision(\"high\")\n\n"
        "DEVICE = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
        "AMP_DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16"
    )
    new_dev_block = (
        "if torch.cuda.is_available():\n"
        "    torch.backends.cudnn.enabled = False\n"
        "    torch.backends.cudnn.benchmark = False\n"
        "if hasattr(torch, \"set_float32_matmul_precision\"):\n"
        "    torch.set_float32_matmul_precision(\"high\")\n\n"
        "DEVICE, AMP_DTYPE = select_compute_device()\n"
        "print(f\"[DEVICE] Активное устройство вычислений: {DEVICE} (AMP={AMP_DTYPE})\")"
    )
    if old_dev_block not in src:
        raise RuntimeError("Не удалось найти блок инициализации DEVICE в Ячейке 2.")
    src = src.replace(old_dev_block, new_dev_block, 1)

    # 3. Безопасный autocast для CPU и Apple Silicon MPS
    src = src.replace(
        'with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_bf16):',
        "with amp_autocast(DEVICE, AMP_DTYPE):",
    )

    # 4. Учитываем выбранный DEVICE в MC Dropout (если пользователь задал --device cpu)
    src = src.replace(
        "if not torch.cuda.is_available() or n_passes < 2:",
        'if DEVICE.type != "cuda" or n_passes < 2:',
    )

    # 5. Добавляем кириллические шрифты Windows в _get_cyrillic_font
    src = src.replace(
        '"/System/Library/Fonts/Supplemental/Arial.ttf",\n    ]',
        '"/System/Library/Fonts/Supplemental/Arial.ttf",\n    ] + get_windows_font_candidates()',
    )

    # 6. Unicode-безопасное сохранение PNG для DOCX на Windows
    src = src.replace(
        "cv2.imwrite(str(img_p), cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))",
        "imwrite_unicode(img_p, cv2.cvtColor(composite_rgb, cv2.COLOR_RGB2BGR))",
    )

    # 7. Локальный запуск Gradio (без inline=True и share=True по умолчанию)
    src = src.replace(
        "demo.queue().launch(share=True, inline=True, debug=False)",
        (
            "demo.queue().launch(\n"
            '        server_name=os.environ.get("CDSS_HOST", "127.0.0.1"),\n'
            '        server_port=int(os.environ.get("CDSS_PORT", "7860")),\n'
            '        share=os.environ.get("CDSS_SHARE", "0") == "1",\n'
            '        inbrowser=os.environ.get("CDSS_NO_BROWSER", "0") != "1",\n'
            "        debug=False,\n"
            "    )"
        ),
    )

    return CORE_HEADER + "\n" + src


def patch_cell3(src: str) -> str:
    # 1. Безопасный autocast
    src = src.replace(
        'with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE, enabled=use_bf16):',
        "with amp_autocast(DEVICE, AMP_DTYPE):",
    )

    # 2. Unicode-безопасное чтение изображений OpenCV на Windows
    src = src.replace(
        "cv2.imread(f_path, cv2.IMREAD_COLOR)",
        "imread_unicode(f_path, cv2.IMREAD_COLOR)",
    )

    # 3. Кроссплатформенные пути кэша и временных файлов
    src = src.replace(
        'Path("/root/.cache/kagglehub/datasets")',
        'Path.home() / ".cache" / "kagglehub" / "datasets"',
    )
    src = src.replace(
        'Path("/tmp")',
        "Path(tempfile.gettempdir())",
    )
    src = src.replace(
        'out_prefix = str(dl_dir) + "/"',
        "out_prefix = str(dl_dir) + os.sep",
    )

    # 4. Поле активной ссылки на интерфейс в шапке (с кнопкой копирования)
    header_anchor = (
        '        gr.Markdown(\n'
        '            "## 🧠 Нейроонкологическая СППР МРТ головного мозга\\n"\n'
        '            f"> *{AI_DISCLAIMER_HEADER}*"\n'
        '        )'
    )
    header_with_link = (
        header_anchor
        + '\n'
        + '        public_link_box = make_copyable_textbox(\n'
        + '            gr,\n'
        + '            label="Адрес этого интерфейса (нажмите на значок копирования справа, чтобы открыть на другом ПК)",\n'
        + '            interactive=False,\n'
        + '            lines=1\n'
        + '        )\n'
        + '        def _get_active_app_url():\n'
        + '            share_u = getattr(demo, "share_url", None)\n'
        + '            if share_u:\n'
        + '                return str(share_u)\n'
        + '            host = os.environ.get("CDSS_HOST", "127.0.0.1")\n'
        + '            port = os.environ.get("CDSS_PORT", "7860")\n'
        + '            return f"http://{host}:{port}"\n'
        + '        demo.load(fn=_get_active_app_url, inputs=[], outputs=[public_link_box])'
    )
    src = src.replace(header_anchor, header_with_link, 1)

    # 5. Быстрая вставка ссылки по умолчанию во вкладке «Демо-плашка 20 МРТ»
    demo20_box_anchor = (
        '                        demo20_url_box = gr.Textbox(\n'
        '                            value=(PUBLIC_DEMO_20_GDRIVE_URL or "").strip(),\n'
        '                            label="Публичная ссылка Google Диска на папку или .zip-архив с 20 файлами МРТ",\n'
        '                            placeholder="https://drive.google.com/drive/folders/... или ссылка на .zip-архив",\n'
        '                            lines=1\n'
        '                        )'
    )
    demo20_box_new = (
        '                        with gr.Row():\n'
        '                            demo20_url_box = gr.Textbox(\n'
        '                                value=(PUBLIC_DEMO_20_GDRIVE_URL or "").strip(),\n'
        '                                label="Публичная ссылка Google Диска на папку или .zip-архив с 20 файлами МРТ",\n'
        '                                placeholder="https://drive.google.com/drive/folders/... или ссылка на .zip-архив",\n'
        '                                lines=1,\n'
        '                                scale=4\n'
        '                            )\n'
        '                            demo20_preset_btn = gr.Button("Вставить ссылку по умолчанию", variant="secondary", scale=1)\n'
        '                        demo20_preset_btn.click(\n'
        '                            fn=lambda: (PUBLIC_DEMO_20_GDRIVE_URL or "").strip(),\n'
        '                            inputs=[],\n'
        '                            outputs=[demo20_url_box]\n'
        '                        )'
    )
    src = src.replace(demo20_box_anchor, demo20_box_new, 1)

    # 6. Быстрая вставка ссылки на снимок МРТ во вкладке «Клинический разбор 1 среза»
    img_input_anchor = '                        img_input = gr.Image(type="numpy", label="Исходный срез МРТ", height=300)'
    img_input_new = (
        '                        with gr.Row():\n'
        '                            single_url_box = gr.Textbox(\n'
        '                                label="Быстрая вставка ссылки на снимок МРТ (прямая ссылка или Google Диск)",\n'
        '                                placeholder="Вставьте ссылку на .jpg / .png / .dcm и нажмите Enter или кнопку справа",\n'
        '                                lines=1,\n'
        '                                scale=4\n'
        '                            )\n'
        '                            single_url_btn = gr.Button("Загрузить и проанализировать", variant="primary", scale=1)\n'
        + img_input_anchor
    )
    src = src.replace(img_input_anchor, img_input_new, 1)

    analyze_click_anchor = (
        '                analyze_btn.click(\n'
        '                    fn=on_analyze,\n'
        '                    inputs=[img_input, seg_slider, unc_slider, spacing_slider, session_state],\n'
        '                    outputs=[session_state, result_preview, canvas_editor, telemetry_md, protocol_box]\n'
        '                )'
    )
    analyze_click_new = (
        analyze_click_anchor
        + '\n'
        + '                def on_analyze_from_url(url_str, seg_t, unc_t, px_sp, state):\n'
        + '                    img_rgb, msg = load_image_from_url(url_str)\n'
        + '                    if img_rgb is None:\n'
        + '                        return None, state, None, None, f"### Ошибка загрузки ссылки\\n{msg}", ""\n'
        + '                    st_out, comp_out, ed_out, tel_out, proto_out = on_analyze(img_rgb, seg_t, unc_t, px_sp, state)\n'
        + '                    return img_rgb, st_out, comp_out, ed_out, tel_out, proto_out\n'
        + '                single_url_btn.click(\n'
        + '                    fn=on_analyze_from_url,\n'
        + '                    inputs=[single_url_box, seg_slider, unc_slider, spacing_slider, session_state],\n'
        + '                    outputs=[img_input, session_state, result_preview, canvas_editor, telemetry_md, protocol_box]\n'
        + '                )\n'
        + '                single_url_box.submit(\n'
        + '                    fn=on_analyze_from_url,\n'
        + '                    inputs=[single_url_box, seg_slider, unc_slider, spacing_slider, session_state],\n'
        + '                    outputs=[img_input, session_state, result_preview, canvas_editor, telemetry_md, protocol_box]\n'
        + '                )'
    )
    src = src.replace(analyze_click_anchor, analyze_click_new, 1)

    # 7. Локальный запуск Gradio с параметрами терминала
    src = src.replace(
        "demo.queue().launch(share=True, inline=True, debug=False)",
        (
            "demo.queue().launch(\n"
            '        server_name=os.environ.get("CDSS_HOST", "127.0.0.1"),\n'
            '        server_port=int(os.environ.get("CDSS_PORT", "7860")),\n'
            '        share=os.environ.get("CDSS_SHARE", "0") == "1",\n'
            '        inbrowser=os.environ.get("CDSS_NO_BROWSER", "0") != "1",\n'
            "        debug=False,\n"
            "    )"
        ),
    )

    return UI_HEADER + "\n" + src

def main() -> None:
    if len(sys.argv) < 2:
        print("Использование: python build_from_notebook.py <путь_к_блокноту.ipynb>")
        sys.exit(1)

    nb_path = Path(sys.argv[1]).expanduser().resolve()
    if not nb_path.is_file():
        print(f"[ERROR] Файл не найден: {nb_path}")
        sys.exit(1)

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    cell2_src = None
    cell3_src = None

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        code = "".join(cell.get("source", []))
        if "class StandaloneL4CDSSController" in code:
            cell2_src = code
        elif "def launch_interactive_cdss_with_batch_auditor_v8" in code:
            cell3_src = code

    if not cell2_src or not cell3_src:
        raise RuntimeError("Не найдены Ячейка 2 или Ячейка 3 в указанном блокноте.")

    out_dir = Path(__file__).resolve().parent
    core_path = out_dir / "cdss_core.py"
    ui_path = out_dir / "cdss_ui.py"

    core_path.write_text(patch_cell2(cell2_src), encoding="utf-8")
    ui_path.write_text(patch_cell3(cell3_src), encoding="utf-8")

    # Проверяем корректность синтаксиса обоих сгенерированных модулей
    py_compile.compile(str(core_path), doraise=True)
    py_compile.compile(str(ui_path), doraise=True)

    print(f"[OK] Сгенерирован и проверен: {core_path}")
    print(f"[OK] Сгенерирован и проверен: {ui_path}")
    print("Теперь можно запускать проект командой: python main.py")


if __name__ == "__main__":
    main()
