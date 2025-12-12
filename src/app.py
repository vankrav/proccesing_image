import io
import json
import os
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rembg import remove

# Импортируем функции из main.py
try:
    from .main import (
        apply_color_reference,
        detect_face_mediapipe,
        expand_to_fill_width,
        align_eyes_vertical,
        fill_bottom_gap_with_last_pixels,
        enhance_face_gfpgan,
        fit_to_size,
        load_image,
        refine_alpha,
        strip_background,
    )
except ImportError:
    # Если запускается напрямую
    from main import (
        apply_color_reference,
        detect_face_mediapipe,
        expand_to_fill_width,
        align_eyes_vertical,
        fill_bottom_gap_with_last_pixels,
        enhance_face_gfpgan,
        fit_to_size,
        load_image,
        refine_alpha,
        strip_background,
    )

app = FastAPI(title="Portrait Processing API", version="1.0.0")

# Создаем папки для временных файлов и результатов
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
CONFIG_FILE = Path("config.json")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def load_default_config() -> dict:
    """Загружает настройки по умолчанию из конфигурационного файла."""
    default_config = {
        "face_enhance": True,
        "face_upscale": 4,
        "face_strength": 0.7,
        "face_iterations": 2,
        "center_face": False,
        "normalize_exposure": False,
        "color_strength": 1.0,
        "reduce_contrast": 0.85,
        "brightness_adjust": 0.0,
        "saturation_adjust": 0.0,
        "sepia_strength": 0.0,
        "keep_largest": True,
        "alpha_erode": 17,
        "alpha_dilate": 0,
        "alpha_feather": 16,
        "bg_model": "u2net_human_seg",
    }
    
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                # Обновляем только те ключи, которые есть в сохраненном конфиге
                default_config.update(saved_config)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Ошибка загрузки конфига: {e}, используем настройки по умолчанию")
    
    return default_config


def save_default_config(config: dict) -> None:
    """Сохраняет настройки в конфигурационный файл."""
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    except IOError as e:
        raise HTTPException(status_code=500, detail=f"Ошибка сохранения конфига: {e}")


def process_image_bytes(
    portrait_bytes: bytes,
    reference_bytes: bytes,
    background_bytes: bytes | None = None,
    face_enhance: bool = True,
    face_upscale: int = 4,
    face_strength: float = 0.7,
    face_iterations: int = 2,
    alpha_erode: int = 17,
    alpha_dilate: int = 0,
    alpha_feather: int = 16,
    bg_model: str = "u2net_human_seg",
    keep_largest: bool = True,
    face_detect: bool = True,
    center_face: bool = False,
    normalize_exposure: bool = False,
    color_strength: float = 1.0,
    reduce_contrast: float = 0.85,
    brightness_adjust: float = 0.0,
    saturation_adjust: float = 0.0,
    sepia_strength: float = 0.0,
) -> Tuple[Image.Image, float]:
    """
    Обрабатывает изображения из байтов и возвращает результат и время обработки.
    Returns: (result_image, elapsed_time_in_seconds)
    """
    start_time = time.time()
    
    # Загружаем изображения из байтов
    portrait = Image.open(io.BytesIO(portrait_bytes))
    # Исправляем ориентацию перед конвертацией
    try:
        from main import fix_image_orientation, normalize_exposure
    except ImportError:
        from .main import fix_image_orientation, normalize_exposure
    portrait = fix_image_orientation(portrait)
    portrait = portrait.convert("RGBA")
    # Нормализуем экспозицию входного изображения (если включено)
    if normalize_exposure:
        portrait = normalize_exposure(portrait, debug=face_detect)
    
    ref = Image.open(io.BytesIO(reference_bytes))
    ref = fix_image_orientation(ref)
    ref = ref.convert("RGBA")

    # Распознавание лица с помощью MediaPipe (для дебага и центрирования)
    face_info = None
    if face_detect:
        print("\n" + "🔍" * 30)
        print("НАЧАЛО РАСПОЗНАВАНИЯ ЛИЦА С MEDIAPIPE (Web API)")
        print("🔍" * 30)
        face_info = detect_face_mediapipe(portrait, debug=True)
        if face_info:
            print("✅ Распознавание лица завершено успешно")
        else:
            print("⚠️ Лицо не обнаружено или MediaPipe недоступен")
        print("🔍" * 30 + "\n")

    # Удаляем фон
    portrait_no_bg = strip_background(portrait, model_name=bg_model)

    # Улучшаем маску (keep_largest=True оставляет только главного человека)
    portrait_no_bg = refine_alpha(
        portrait_no_bg,
        erode=alpha_erode,
        dilate=alpha_dilate,
        feather=alpha_feather,
        keep_largest=keep_largest,
    )
    
    # Центрируем лицо по горизонтали (только если включено)
    if center_face:
        try:
            from main import center_face_horizontally
        except ImportError:
            from .main import center_face_horizontally
        # Нужно пересчитать координаты лица для изображения без фона
        face_info_no_bg = detect_face_mediapipe(portrait_no_bg, debug=False)
        if face_info_no_bg:
            if face_detect:
                print("\n🎯 ПРИМЕНЕНИЕ ЦЕНТРИРОВАНИЯ ЛИЦА (Web API)")
            portrait_no_bg = center_face_horizontally(portrait_no_bg, face_info_no_bg, debug=face_detect)
            portrait_no_bg = expand_to_fill_width(portrait_no_bg, padding=0, min_scale=1.12, debug=face_detect)
            face_info_scaled = detect_face_mediapipe(portrait_no_bg, debug=False)
            portrait_no_bg = align_eyes_vertical(portrait_no_bg, face_info_scaled or face_info_no_bg, target_frac=1/3, debug=face_detect)
            portrait_no_bg = fill_bottom_gap_with_last_pixels(portrait_no_bg, debug=face_detect)
            if face_detect:
                print("🎯" * 30 + "\n")
    elif face_detect:
        print("\n⚠️ Лицо не обнаружено на изображении без фона, центрирование пропущено\n")

    # Применяем цвет референса
    colored = apply_color_reference(
        portrait_no_bg,
        ref,
        color_strength=color_strength,
        reduce_contrast=reduce_contrast,
        brightness_adjust=brightness_adjust,
        saturation_adjust=saturation_adjust,
    )

    # Улучшаем лицо (если нужно)
    if face_enhance:
        colored = enhance_face_gfpgan(
            colored,
            upscale=face_upscale,
            strength=face_strength,
            iterations=face_iterations,
        )

    # Подкладываем фон (если есть)
    if background_bytes:
        bg = Image.open(io.BytesIO(background_bytes)).convert("RGBA")
        bg_resized = bg.resize(colored.size, Image.LANCZOS)
        colored = Image.alpha_composite(bg_resized, colored)

    # Вписываем изображение в размер 720x1280 с сохранением пропорций
    # anchor_y=1/3 чтобы линия глаз оставалась ближе к верхней трети
    colored = fit_to_size(colored, (720, 1280), anchor_y=1/3)
    
    # Накладываем сепию (если включено)
    if sepia_strength > 0:
        try:
            from main import apply_sepia
        except ImportError:
            from .main import apply_sepia
        colored = apply_sepia(colored, strength=sepia_strength)

    elapsed_time = time.time() - start_time
    return colored, elapsed_time


@app.get("/", response_class=HTMLResponse)
async def root():
    """Главная страница с веб-интерфейсом."""
    html_content = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portrait Processing - Web Interface</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            opacity: 0.9;
        }
        .content {
            padding: 40px;
        }
        .form-group {
            margin-bottom: 25px;
        }
        .form-group label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        .file-input-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        .file-input-wrapper input[type="file"] {
            width: 100%;
            padding: 12px;
            border: 2px dashed #667eea;
            border-radius: 10px;
            background: #f8f9fa;
            cursor: pointer;
            transition: all 0.3s;
        }
        .file-input-wrapper input[type="file"]:hover {
            border-color: #764ba2;
            background: #e9ecef;
        }
        .preview {
            margin-top: 15px;
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .preview img {
            width: 100%;
            border-radius: 10px;
            border: 2px solid #e9ecef;
            max-height: 200px;
            object-fit: contain;
            background: #f8f9fa;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }
        .control-group {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        .control-group h3 {
            margin-bottom: 15px;
            color: #667eea;
        }
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
        }
        .checkbox-group input[type="checkbox"] {
            width: 20px;
            height: 20px;
            cursor: pointer;
        }
        .range-group {
            margin-bottom: 15px;
        }
        .range-group input[type="range"] {
            width: 100%;
            margin-top: 5px;
        }
        .range-value {
            display: inline-block;
            margin-left: 10px;
            font-weight: bold;
            color: #667eea;
        }
        .preset-buttons {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        .preset-btn {
            flex: 1;
            padding: 10px;
            border: 2px solid #667eea;
            background: white;
            color: #667eea;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s;
        }
        .preset-btn:hover {
            background: #667eea;
            color: white;
        }
        .submit-btn {
            width: 100%;
            padding: 18px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: 600;
            cursor: pointer;
            margin-top: 30px;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .submit-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .submit-btn:active {
            transform: translateY(0);
        }
        .submit-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }
        .result {
            margin-top: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            text-align: center;
        }
        .result img {
            max-width: 100%;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        .loading.active {
            display: block;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }
        .error.active {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 Portrait Processing</h1>
            <p>Обработка портретов: удаление фона, перенос цвета, улучшение лица</p>
        </div>
        <div class="content">
            <form id="processForm" enctype="multipart/form-data">
                <div class="form-group">
                    <label>Портрет (исходное изображение)</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="portrait" name="portrait" accept="image/*" required>
                    </div>
                    <div class="preview" id="portraitPreview"></div>
                </div>

                <div class="form-group">
                    <label>Референс (изображение для переноса цвета) <small style="color: #666;">(по умолчанию: src/ref.png)</small></label>
                    <div class="file-input-wrapper">
                        <input type="file" id="reference" name="reference" accept="image/*">
                    </div>
                    <div class="preview" id="referencePreview"></div>
                </div>

                <div class="form-group">
                    <label>Фон <small style="color: #666;">(по умолчанию: src/bg.jpg)</small></label>
                    <div class="file-input-wrapper">
                        <input type="file" id="background" name="background" accept="image/*">
                    </div>
                    <div class="preview" id="backgroundPreview"></div>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <h3>Улучшение лица</h3>
                        <div class="checkbox-group">
                            <input type="checkbox" id="faceEnhance" name="face_enhance" checked>
                            <label for="faceEnhance">Включить GFPGAN</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="centerFace" name="center_face">
                            <label for="centerFace">Центрировать лицо (горизонтально и вертикально)</label>
                        </div>
                        <div class="checkbox-group">
                            <input type="checkbox" id="normalizeExposure" name="normalize_exposure">
                            <label for="normalizeExposure">Нормализовать экспозицию (увеличить яркость)</label>
                        </div>
                        <div class="range-group">
                            <label>Масштаб улучшения: <span class="range-value" id="upscaleValue">4</span></label>
                            <input type="range" id="faceUpscale" name="face_upscale" min="1" max="4" value="4" step="1">
                        </div>
                        <div class="range-group">
                            <label>Интенсивность: <span class="range-value" id="strengthValue">0.7</span></label>
                            <input type="range" id="faceStrength" name="face_strength" min="0" max="1" value="0.7" step="0.1">
                        </div>
                        <div class="range-group">
                            <label>Итерации: <span class="range-value" id="iterationsValue">2</span></label>
                            <input type="range" id="faceIterations" name="face_iterations" min="1" max="3" value="2" step="1">
                        </div>
                    </div>

                    <div class="control-group">
                        <h3>Настройка цвета референса</h3>
                        <div class="range-group">
                            <label>Интенсивность переноса: <span class="range-value" id="colorStrengthValue">1.0</span></label>
                            <input type="range" id="colorStrength" name="color_strength" min="0" max="1" value="1.0" step="0.1">
                        </div>
                        <div class="range-group">
                            <label>Снижение контраста: <span class="range-value" id="reduceContrastValue">0.85</span></label>
                            <input type="range" id="reduceContrast" name="reduce_contrast" min="0.5" max="1.0" value="0.85" step="0.05">
                        </div>
                        <div class="range-group">
                            <label>Коррекция яркости: <span class="range-value" id="brightnessAdjustValue">0.0</span></label>
                            <input type="range" id="brightnessAdjust" name="brightness_adjust" min="-1" max="1" value="0.0" step="0.1">
                        </div>
                        <div class="range-group">
                            <label>Коррекция насыщенности: <span class="range-value" id="saturationAdjustValue">0.0</span></label>
                            <input type="range" id="saturationAdjust" name="saturation_adjust" min="-1" max="1" value="0.0" step="0.1">
                        </div>
                        <div class="range-group">
                            <label>Эффект сепии: <span class="range-value" id="sepiaStrengthValue">0.0</span></label>
                            <input type="range" id="sepiaStrength" name="sepia_strength" min="0" max="1" value="0.0" step="0.1">
                        </div>
                    </div>

                    <div class="control-group">
                        <h3>Очистка краев</h3>
                        <div class="checkbox-group">
                            <input type="checkbox" id="keepLargest" name="keep_largest" checked>
                            <label for="keepLargest">Оставить только главного человека (убрать людей на фоне)</label>
                        </div>
                        <div class="range-group">
                            <label>Эрозия: <span class="range-value" id="erodeValue">17</span></label>
                            <input type="range" id="alphaErode" name="alpha_erode" min="0" max="25" value="17">
                        </div>
                        <div class="range-group">
                            <label>Дилятация: <span class="range-value" id="dilateValue">0</span></label>
                            <input type="range" id="alphaDilate" name="alpha_dilate" min="0" max="10" value="0">
                        </div>
                        <div class="range-group">
                            <label>Размытие: <span class="range-value" id="featherValue">16</span></label>
                            <input type="range" id="alphaFeather" name="alpha_feather" min="0" max="25" value="16">
                        </div>
                    </div>

                    <div class="control-group">
                        <h3>Модель удаления фона</h3>
                        <div class="range-group">
                            <label for="bgModel">Модель:</label>
                            <select id="bgModel" name="bg_model" style="width: 100%; padding: 8px; border: 2px solid #e9ecef; border-radius: 8px; font-size: 1em; margin-top: 5px;">
                                <option value="isnet-general-use">ISNet (лучшая точность) ⭐</option>
                                <option value="u2net_human_seg" selected>U2Net Human Seg (для людей)</option>
                                <option value="u2net">U2Net (базовая, быстрая)</option>
                                <option value="silueta">Silueta (универсальная)</option>
                                <option value="u2netp">U2Net Lite (самая быстрая)</option>
                            </select>
                        </div>
                    </div>
                </div>

                <div class="preset-buttons">
                    <button type="button" class="preset-btn" onclick="applyPreset('face3')">Пресет Face3</button>
                    <button type="button" class="preset-btn" onclick="applyPreset('face8')">Пресет Face8</button>
                </div>

                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button type="button" class="preset-btn" onclick="saveConfig()" style="background: #28a745; color: white; border: none;">
                        💾 Сохранить настройки по умолчанию
                    </button>
                    <button type="button" class="preset-btn" onclick="loadConfig()" style="background: #17a2b8; color: white; border: none;">
                        📂 Загрузить сохраненные настройки
                    </button>
                </div>

                <button type="submit" class="submit-btn" id="submitBtn">Обработать изображение</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Обработка изображения...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="result" id="result" style="display: none;">
                <h2>Результат</h2>
                <p id="processingTime" style="color: #667eea; font-weight: 600; margin-bottom: 15px; font-size: 1.1em; padding: 10px; background: #f0f4ff; border-radius: 8px; display: inline-block;">⏱️ Время обработки: вычисляется...</p>
                <img id="resultImage" alt="Результат обработки" style="margin-top: 15px;">
                <div style="margin-top: 20px;">
                    <a id="downloadLink" download="result.png" style="display: inline-block; padding: 12px 24px; background: #667eea; color: white; text-decoration: none; border-radius: 8px; font-weight: 600;">Скачать результат</a>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Превью изображений
        document.getElementById('portrait').addEventListener('change', function(e) {
            previewImage(e.target, 'portraitPreview');
        });
        document.getElementById('reference').addEventListener('change', function(e) {
            previewImage(e.target, 'referencePreview');
        });
        document.getElementById('background').addEventListener('change', function(e) {
            previewImage(e.target, 'backgroundPreview');
        });

        function previewImage(input, previewId) {
            const preview = document.getElementById(previewId);
            preview.innerHTML = '';
            if (input.files && input.files[0]) {
                const reader = new FileReader();
                reader.onload = function(e) {
                    const img = document.createElement('img');
                    img.src = e.target.result;
                    preview.appendChild(img);
                };
                reader.readAsDataURL(input.files[0]);
            }
        }

        // Обновление значений слайдеров
        document.getElementById('faceUpscale').addEventListener('input', function(e) {
            document.getElementById('upscaleValue').textContent = e.target.value;
        });
        document.getElementById('faceStrength').addEventListener('input', function(e) {
            document.getElementById('strengthValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        document.getElementById('faceIterations').addEventListener('input', function(e) {
            document.getElementById('iterationsValue').textContent = e.target.value;
        });
        document.getElementById('alphaErode').addEventListener('input', function(e) {
            document.getElementById('erodeValue').textContent = e.target.value;
        });
        document.getElementById('alphaDilate').addEventListener('input', function(e) {
            document.getElementById('dilateValue').textContent = e.target.value;
        });
        document.getElementById('alphaFeather').addEventListener('input', function(e) {
            document.getElementById('featherValue').textContent = e.target.value;
        });
        document.getElementById('colorStrength').addEventListener('input', function(e) {
            document.getElementById('colorStrengthValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        document.getElementById('reduceContrast').addEventListener('input', function(e) {
            document.getElementById('reduceContrastValue').textContent = parseFloat(e.target.value).toFixed(2);
        });
        document.getElementById('brightnessAdjust').addEventListener('input', function(e) {
            document.getElementById('brightnessAdjustValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        document.getElementById('saturationAdjust').addEventListener('input', function(e) {
            document.getElementById('saturationAdjustValue').textContent = parseFloat(e.target.value).toFixed(1);
        });
        document.getElementById('sepiaStrength').addEventListener('input', function(e) {
            document.getElementById('sepiaStrengthValue').textContent = parseFloat(e.target.value).toFixed(1);
        });

        // Загрузка сохраненных настроек при загрузке страницы
        async function loadConfig() {
            try {
                const response = await fetch('/api/config');
                if (!response.ok) {
                    throw new Error('Ошибка загрузки настроек');
                }
                const config = await response.json();
                
                // Применяем настройки к форме
                if (config.face_enhance !== undefined) document.getElementById('faceEnhance').checked = config.face_enhance;
                if (config.face_upscale !== undefined) {
                    document.getElementById('faceUpscale').value = config.face_upscale;
                    document.getElementById('upscaleValue').textContent = config.face_upscale;
                }
                if (config.face_strength !== undefined) {
                    document.getElementById('faceStrength').value = config.face_strength;
                    document.getElementById('strengthValue').textContent = parseFloat(config.face_strength).toFixed(1);
                }
                if (config.face_iterations !== undefined) {
                    document.getElementById('faceIterations').value = config.face_iterations;
                    document.getElementById('iterationsValue').textContent = config.face_iterations;
                }
                if (config.center_face !== undefined) document.getElementById('centerFace').checked = config.center_face;
                if (config.normalize_exposure !== undefined) document.getElementById('normalizeExposure').checked = config.normalize_exposure;
                if (config.color_strength !== undefined) {
                    document.getElementById('colorStrength').value = config.color_strength;
                    document.getElementById('colorStrengthValue').textContent = parseFloat(config.color_strength).toFixed(1);
                }
                if (config.reduce_contrast !== undefined) {
                    document.getElementById('reduceContrast').value = config.reduce_contrast;
                    document.getElementById('reduceContrastValue').textContent = parseFloat(config.reduce_contrast).toFixed(2);
                }
                if (config.brightness_adjust !== undefined) {
                    document.getElementById('brightnessAdjust').value = config.brightness_adjust;
                    document.getElementById('brightnessAdjustValue').textContent = parseFloat(config.brightness_adjust).toFixed(1);
                }
                if (config.saturation_adjust !== undefined) {
                    document.getElementById('saturationAdjust').value = config.saturation_adjust;
                    document.getElementById('saturationAdjustValue').textContent = parseFloat(config.saturation_adjust).toFixed(1);
                }
                if (config.sepia_strength !== undefined) {
                    document.getElementById('sepiaStrength').value = config.sepia_strength;
                    document.getElementById('sepiaStrengthValue').textContent = parseFloat(config.sepia_strength).toFixed(1);
                }
                if (config.keep_largest !== undefined) document.getElementById('keepLargest').checked = config.keep_largest;
                if (config.alpha_erode !== undefined) {
                    document.getElementById('alphaErode').value = config.alpha_erode;
                    document.getElementById('erodeValue').textContent = config.alpha_erode;
                }
                if (config.alpha_dilate !== undefined) {
                    document.getElementById('alphaDilate').value = config.alpha_dilate;
                    document.getElementById('dilateValue').textContent = config.alpha_dilate;
                }
                if (config.alpha_feather !== undefined) {
                    document.getElementById('alphaFeather').value = config.alpha_feather;
                    document.getElementById('featherValue').textContent = config.alpha_feather;
                }
                if (config.bg_model !== undefined) document.getElementById('bgModel').value = config.bg_model;
                
                alert('✅ Настройки загружены!');
            } catch (error) {
                alert('❌ Ошибка загрузки настроек: ' + error.message);
            }
        }

        // Сохранение текущих настроек
        async function saveConfig() {
            const formData = new FormData();
            formData.append('face_enhance', document.getElementById('faceEnhance').checked);
            formData.append('face_upscale', document.getElementById('faceUpscale').value);
            formData.append('face_strength', document.getElementById('faceStrength').value);
            formData.append('face_iterations', document.getElementById('faceIterations').value);
            formData.append('center_face', document.getElementById('centerFace').checked);
            formData.append('normalize_exposure', document.getElementById('normalizeExposure').checked);
            formData.append('color_strength', document.getElementById('colorStrength').value);
            formData.append('reduce_contrast', document.getElementById('reduceContrast').value);
            formData.append('brightness_adjust', document.getElementById('brightnessAdjust').value);
            formData.append('saturation_adjust', document.getElementById('saturationAdjust').value);
            formData.append('sepia_strength', document.getElementById('sepiaStrength').value);
            formData.append('keep_largest', document.getElementById('keepLargest').checked);
            formData.append('alpha_erode', document.getElementById('alphaErode').value);
            formData.append('alpha_dilate', document.getElementById('alphaDilate').value);
            formData.append('alpha_feather', document.getElementById('alphaFeather').value);
            formData.append('bg_model', document.getElementById('bgModel').value);

            try {
                const response = await fetch('/api/config', {
                    method: 'POST',
                    body: formData
                });
                if (!response.ok) {
                    throw new Error('Ошибка сохранения настроек');
                }
                alert('✅ Настройки сохранены!');
            } catch (error) {
                alert('❌ Ошибка сохранения настроек: ' + error.message);
            }
        }

        // Автоматически загружаем настройки при загрузке страницы
        window.addEventListener('DOMContentLoaded', function() {
            loadConfig();
        });

        // Применение пресетов
        function applyPreset(preset) {
            if (preset === 'face3') {
                document.getElementById('faceEnhance').checked = true;
                document.getElementById('faceUpscale').value = 3;
                document.getElementById('upscaleValue').textContent = '3';
                document.getElementById('faceStrength').value = 1.0;
                document.getElementById('strengthValue').textContent = '1.0';
                document.getElementById('faceIterations').value = 1;
                document.getElementById('iterationsValue').textContent = '1';
            } else if (preset === 'face8') {
                document.getElementById('faceEnhance').checked = true;
                document.getElementById('faceUpscale').value = 4;
                document.getElementById('upscaleValue').textContent = '4';
                document.getElementById('faceStrength').value = 1.0;
                document.getElementById('strengthValue').textContent = '1.0';
                document.getElementById('faceIterations').value = 2;
                document.getElementById('iterationsValue').textContent = '2';
            }
        }

        // Отправка формы
        document.getElementById('processForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData();
            formData.append('portrait', document.getElementById('portrait').files[0]);
            
            const referenceFile = document.getElementById('reference').files[0];
            if (referenceFile) {
                formData.append('reference', referenceFile);
            }
            
            const backgroundFile = document.getElementById('background').files[0];
            if (backgroundFile) {
                formData.append('background', backgroundFile);
            }

            formData.append('face_enhance', document.getElementById('faceEnhance').checked);
            formData.append('center_face', document.getElementById('centerFace').checked);
            formData.append('normalize_exposure', document.getElementById('normalizeExposure').checked);
            formData.append('face_upscale', document.getElementById('faceUpscale').value);
            formData.append('face_strength', document.getElementById('faceStrength').value);
            formData.append('face_iterations', document.getElementById('faceIterations').value);
            formData.append('keep_largest', document.getElementById('keepLargest').checked);
            formData.append('alpha_erode', document.getElementById('alphaErode').value);
            formData.append('alpha_dilate', document.getElementById('alphaDilate').value);
            formData.append('alpha_feather', document.getElementById('alphaFeather').value);
            formData.append('color_strength', document.getElementById('colorStrength').value);
            formData.append('reduce_contrast', document.getElementById('reduceContrast').value);
            formData.append('brightness_adjust', document.getElementById('brightnessAdjust').value);
            formData.append('saturation_adjust', document.getElementById('saturationAdjust').value);
            formData.append('sepia_strength', document.getElementById('sepiaStrength').value);
            formData.append('bg_model', document.getElementById('bgModel').value);

            // Показываем загрузку
            document.getElementById('loading').classList.add('active');
            document.getElementById('result').style.display = 'none';
            document.getElementById('error').classList.remove('active');
            document.getElementById('submitBtn').disabled = true;

            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Ошибка обработки');
                }

                // Получаем время обработки из заголовка ДО вызова blob()
                const processingTime = response.headers.get('X-Processing-Time') || 'N/A';
                
                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                document.getElementById('resultImage').src = imageUrl;
                document.getElementById('downloadLink').href = imageUrl;
                document.getElementById('processingTime').textContent = `⏱️ Время обработки: ${processingTime} сек`;
                document.getElementById('result').style.display = 'block';
            } catch (error) {
                document.getElementById('error').textContent = 'Ошибка: ' + error.message;
                document.getElementById('error').classList.add('active');
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('submitBtn').disabled = false;
            }
        });
    </script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/process")
async def process_image(
    portrait: UploadFile = File(...),
    reference: UploadFile = File(None),
    background: UploadFile = File(None),
    face_enhance: bool = Form(True),
    face_upscale: int = Form(4),
    face_strength: float = Form(0.7),
    face_iterations: int = Form(2),
    alpha_erode: int = Form(17),
    alpha_dilate: int = Form(0),
    alpha_feather: int = Form(16),
    bg_model: str = Form("u2net_human_seg"),
    keep_largest: bool = Form(True),
    face_detect: bool = Form(True),
    center_face: bool = Form(False),
    normalize_exposure: bool = Form(False),
    color_strength: float = Form(1.0),
    reduce_contrast: float = Form(0.85),
    brightness_adjust: float = Form(0.0),
    saturation_adjust: float = Form(0.0),
    sepia_strength: float = Form(0.0),
):
    """API endpoint для обработки изображений."""
    try:
        # Валидация модели
        valid_models = ["isnet-general-use", "u2net_human_seg", "u2net", "silueta", "u2netp"]
        if bg_model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Неверная модель. Доступны: {', '.join(valid_models)}")
        
        # Читаем файлы
        portrait_bytes = await portrait.read()
        
        # Используем src/ref.png по умолчанию, если референс не загружен
        if reference and reference.filename:
            reference_bytes = await reference.read()
        else:
            # Пытаемся загрузить src/ref.png по умолчанию
            ref_path = Path("src/ref.png")
            if ref_path.exists():
                with open(ref_path, "rb") as f:
                    reference_bytes = f.read()
            else:
                raise HTTPException(status_code=400, detail="Референс не указан и src/ref.png не найден")
        
        # Используем src/bg.jpg по умолчанию, если фон не загружен
        background_bytes = None
        if background and background.filename:
            background_bytes = await background.read()
        else:
            # Пытаемся загрузить src/bg.jpg по умолчанию
            bg_path = Path("src/bg.jpg")
            if bg_path.exists():
                with open(bg_path, "rb") as f:
                    background_bytes = f.read()

        # Обрабатываем изображение
        result_image, elapsed_time = process_image_bytes(
            portrait_bytes=portrait_bytes,
            reference_bytes=reference_bytes,
            background_bytes=background_bytes,
            face_enhance=face_enhance,
            face_upscale=face_upscale,
            face_strength=face_strength,
            face_iterations=face_iterations,
            alpha_erode=alpha_erode,
            alpha_dilate=alpha_dilate,
            alpha_feather=alpha_feather,
            bg_model=bg_model,
            keep_largest=keep_largest,
            face_detect=face_detect,
            center_face=center_face,
            normalize_exposure=normalize_exposure,
            color_strength=color_strength,
            reduce_contrast=reduce_contrast,
            brightness_adjust=brightness_adjust,
            saturation_adjust=saturation_adjust,
            sepia_strength=sepia_strength,
        )

        # Конвертируем в байты
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        # Добавляем время обработки в заголовок ответа
        return StreamingResponse(
            io.BytesIO(img_bytes.read()),
            media_type="image/png",
            headers={
                "Content-Disposition": "attachment; filename=result.png",
                "X-Processing-Time": f"{elapsed_time:.2f}",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Проверка работоспособности API."""
    return {"status": "ok", "message": "API is running"}


@app.get("/api/config")
async def get_config():
    """Получить сохраненные настройки по умолчанию."""
    config = load_default_config()
    return JSONResponse(content=config)


@app.post("/api/config")
async def save_config(
    face_enhance: bool = Form(True),
    face_upscale: int = Form(4),
    face_strength: float = Form(0.7),
    face_iterations: int = Form(2),
    center_face: bool = Form(False),
    normalize_exposure: bool = Form(False),
    color_strength: float = Form(1.0),
    reduce_contrast: float = Form(0.85),
    brightness_adjust: float = Form(0.0),
    saturation_adjust: float = Form(0.0),
    sepia_strength: float = Form(0.0),
    keep_largest: bool = Form(True),
    alpha_erode: int = Form(17),
    alpha_dilate: int = Form(0),
    alpha_feather: int = Form(16),
    bg_model: str = Form("u2net_human_seg"),
):
    """Сохранить настройки в конфигурационный файл."""
    config = {
        "face_enhance": face_enhance,
        "face_upscale": face_upscale,
        "face_strength": face_strength,
        "face_iterations": face_iterations,
        "center_face": center_face,
        "normalize_exposure": normalize_exposure,
        "color_strength": color_strength,
        "reduce_contrast": reduce_contrast,
        "brightness_adjust": brightness_adjust,
        "saturation_adjust": saturation_adjust,
        "sepia_strength": sepia_strength,
        "keep_largest": keep_largest,
        "alpha_erode": alpha_erode,
        "alpha_dilate": alpha_dilate,
        "alpha_feather": alpha_feather,
        "bg_model": bg_model,
    }
    save_default_config(config)
    return JSONResponse(content={"status": "ok", "message": "Настройки сохранены"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

