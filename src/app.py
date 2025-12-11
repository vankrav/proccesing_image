import io
import os
import sys
import tempfile
import types
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from rembg import remove

# Импортируем функции из main.py
try:
    from .main import (
        apply_color_reference,
        enhance_face_gfpgan,
        load_image,
        refine_alpha,
        strip_background,
    )
except ImportError:
    # Если запускается напрямую
    from main import (
        apply_color_reference,
        enhance_face_gfpgan,
        load_image,
        refine_alpha,
        strip_background,
    )

app = FastAPI(title="Portrait Processing API", version="1.0.0")

# Создаем папки для временных файлов и результатов
UPLOAD_DIR = Path("uploads")
RESULT_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def process_image_bytes(
    portrait_bytes: bytes,
    reference_bytes: bytes,
    background_bytes: bytes | None = None,
    face_enhance: bool = False,
    face_upscale: int = 2,
    face_strength: float = 1.0,
    face_iterations: int = 1,
    alpha_erode: int = 0,
    alpha_dilate: int = 0,
    alpha_feather: int = 0,
    bg_model: str = "isnet-general-use",
) -> Image.Image:
    """Обрабатывает изображения из байтов и возвращает результат."""
    # Загружаем изображения из байтов
    portrait = Image.open(io.BytesIO(portrait_bytes)).convert("RGBA")
    ref = Image.open(io.BytesIO(reference_bytes)).convert("RGBA")

    # Удаляем фон
    portrait_no_bg = strip_background(portrait, model_name=bg_model)

    # Улучшаем маску
    portrait_no_bg = refine_alpha(
        portrait_no_bg, erode=alpha_erode, dilate=alpha_dilate, feather=alpha_feather
    )

    # Применяем цвет референса
    colored = apply_color_reference(portrait_no_bg, ref)

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

    return colored


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
                    <label>Референс (изображение для переноса цвета)</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="reference" name="reference" accept="image/*" required>
                    </div>
                    <div class="preview" id="referencePreview"></div>
                </div>

                <div class="form-group">
                    <label>Фон (опционально)</label>
                    <div class="file-input-wrapper">
                        <input type="file" id="background" name="background" accept="image/*">
                    </div>
                    <div class="preview" id="backgroundPreview"></div>
                </div>

                <div class="controls">
                    <div class="control-group">
                        <h3>Улучшение лица</h3>
                        <div class="checkbox-group">
                            <input type="checkbox" id="faceEnhance" name="face_enhance">
                            <label for="faceEnhance">Включить GFPGAN</label>
                        </div>
                        <div class="range-group">
                            <label>Масштаб улучшения: <span class="range-value" id="upscaleValue">2</span></label>
                            <input type="range" id="faceUpscale" name="face_upscale" min="1" max="4" value="2" step="1">
                        </div>
                        <div class="range-group">
                            <label>Интенсивность: <span class="range-value" id="strengthValue">1.0</span></label>
                            <input type="range" id="faceStrength" name="face_strength" min="0" max="1" value="1.0" step="0.1">
                        </div>
                        <div class="range-group">
                            <label>Итерации: <span class="range-value" id="iterationsValue">1</span></label>
                            <input type="range" id="faceIterations" name="face_iterations" min="1" max="3" value="1" step="1">
                        </div>
                    </div>

                    <div class="control-group">
                        <h3>Очистка краев</h3>
                        <div class="range-group">
                            <label>Эрозия: <span class="range-value" id="erodeValue">0</span></label>
                            <input type="range" id="alphaErode" name="alpha_erode" min="0" max="10" value="0">
                        </div>
                        <div class="range-group">
                            <label>Дилятация: <span class="range-value" id="dilateValue">0</span></label>
                            <input type="range" id="alphaDilate" name="alpha_dilate" min="0" max="10" value="0">
                        </div>
                        <div class="range-group">
                            <label>Размытие: <span class="range-value" id="featherValue">0</span></label>
                            <input type="range" id="alphaFeather" name="alpha_feather" min="0" max="10" value="0">
                        </div>
                    </div>

                    <div class="control-group">
                        <h3>Модель удаления фона</h3>
                        <div class="range-group">
                            <label for="bgModel">Модель:</label>
                            <select id="bgModel" name="bg_model" style="width: 100%; padding: 8px; border: 2px solid #e9ecef; border-radius: 8px; font-size: 1em; margin-top: 5px;">
                                <option value="isnet-general-use" selected>ISNet (лучшая точность) ⭐</option>
                                <option value="u2net_human_seg">U2Net Human Seg (для людей)</option>
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

                <button type="submit" class="submit-btn" id="submitBtn">Обработать изображение</button>
            </form>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Обработка изображения...</p>
            </div>

            <div class="error" id="error"></div>

            <div class="result" id="result" style="display: none;">
                <h2>Результат</h2>
                <img id="resultImage" alt="Результат обработки">
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
            formData.append('reference', document.getElementById('reference').files[0]);
            
            const backgroundFile = document.getElementById('background').files[0];
            if (backgroundFile) {
                formData.append('background', backgroundFile);
            }

            formData.append('face_enhance', document.getElementById('faceEnhance').checked);
            formData.append('face_upscale', document.getElementById('faceUpscale').value);
            formData.append('face_strength', document.getElementById('faceStrength').value);
            formData.append('face_iterations', document.getElementById('faceIterations').value);
            formData.append('alpha_erode', document.getElementById('alphaErode').value);
            formData.append('alpha_dilate', document.getElementById('alphaDilate').value);
            formData.append('alpha_feather', document.getElementById('alphaFeather').value);
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

                const blob = await response.blob();
                const imageUrl = URL.createObjectURL(blob);
                
                document.getElementById('resultImage').src = imageUrl;
                document.getElementById('downloadLink').href = imageUrl;
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
    reference: UploadFile = File(...),
    background: UploadFile = File(None),
    face_enhance: bool = Form(False),
    face_upscale: int = Form(2),
    face_strength: float = Form(1.0),
    face_iterations: int = Form(1),
    alpha_erode: int = Form(0),
    alpha_dilate: int = Form(0),
    alpha_feather: int = Form(0),
    bg_model: str = Form("isnet-general-use"),
):
    """API endpoint для обработки изображений."""
    try:
        # Валидация модели
        valid_models = ["isnet-general-use", "u2net_human_seg", "u2net", "silueta", "u2netp"]
        if bg_model not in valid_models:
            raise HTTPException(status_code=400, detail=f"Неверная модель. Доступны: {', '.join(valid_models)}")
        
        # Читаем файлы
        portrait_bytes = await portrait.read()
        reference_bytes = await reference.read()
        background_bytes = None
        if background:
            background_bytes = await background.read()

        # Обрабатываем изображение
        result_image = process_image_bytes(
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
        )

        # Конвертируем в байты
        img_bytes = io.BytesIO()
        result_image.save(img_bytes, format="PNG")
        img_bytes.seek(0)

        return StreamingResponse(
            io.BytesIO(img_bytes.read()),
            media_type="image/png",
            headers={"Content-Disposition": "attachment; filename=result.png"},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health():
    """Проверка работоспособности API."""
    return {"status": "ok", "message": "API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

