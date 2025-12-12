# Portrait Processing CLI

CLI-скрипт для обработки портретов: удаление фона, перенос цвета по референсу, улучшение лица и наложение фона.

## Структура проекта

```
proccesing_image/
├── README.md          # Этот файл
├── .gitignore         # Исключения для Git
└── src/               # Исходный код
    ├── main.py        # Основной скрипт
    ├── requirements.txt
    ├── image.jpg      # Пример входного изображения
    ├── ref.png        # Пример референса по цвету
    └── bg.jpg         # Пример фона
```

## Быстрый старт

### Запуск веб-сервера (рекомендуется)

#### Windows (PowerShell)

```powershell
# 1. Создать виртуальное окружение
python -m venv .venv

# 2. Активировать (если возникает ошибка политики выполнения)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1

# Альтернатива: использовать activate.bat (CMD)
# .venv\Scripts\activate.bat

# 3. Установить зависимости
pip install -r src/requirements.txt

# 4. Запустить веб-сервер
python src/app.py
```

Откройте браузер и перейдите на `http://localhost:8000` для доступа к веб-интерфейсу.

#### Linux/macOS

```bash
# 1. Создать виртуальное окружение
python3 -m venv .venv

# 2. Активировать
source .venv/bin/activate

# 3. Установить зависимости
pip install -r src/requirements.txt

# 4. Запустить веб-сервер
python src/app.py
```

Откройте браузер и перейдите на `http://localhost:8000` для доступа к веб-интерфейсу.

### CLI режим (командная строка)

Если нужно использовать через командную строку:

```bash
# После установки зависимостей (см. выше)
python src/main.py src/image.jpg src/ref.png -o result.png
```

## Основные возможности

- **Веб-интерфейс** — удобный графический интерфейс для обработки изображений
- **REST API** — программный доступ к функциям обработки
- **Удаление фона** — автоматическое удаление фона с помощью rembg (ISNet, U2Net и др.)
- **Перенос цвета** — приведение портрета к цветам референса (метод Рейнхарда в Lab)
- **Улучшение лица** — сглаживание кожи через GFPGAN (опционально)
- **Очистка краев** — эрозия, дилятация и размытие для чистых краев
- **Наложение фона** — подкладывание заданного фона под результат

## Использование

### Веб-интерфейс

1. Запустите сервер: `python src/app.py`
2. Откройте браузер: `http://localhost:8000`
3. Загрузите изображения (портрет, референс, опционально фон)
4. Настройте параметры обработки
5. Нажмите "Обработать изображение"
6. Скачайте результат

### API Endpoints

#### POST `/api/process`
Обработка изображения через API.

**Параметры (multipart/form-data):**
- `portrait` (file, required) — исходное изображение
- `reference` (file, required) — референс по цвету
- `background` (file, optional) — фон
- `face_enhance` (bool, default: false) — включить улучшение лица
- `face_upscale` (int, default: 1) — масштаб GFPGAN (1-4)
- `bg_model` (string, default: "isnet-general-use") — модель удаления фона
- `alpha_erode` (int, default: 0) — эрозия маски
- `alpha_dilate` (int, default: 0) — дилятация маски
- `alpha_feather` (int, default: 0) — размытие края

**Ответ:** PNG изображение

**Пример (curl):**
```bash
curl -X POST "http://localhost:8000/api/process" \
  -F "portrait=@src/image.jpg" \
  -F "reference=@src/ref.png" \
  -F "background=@src/bg.jpg" \
  -F "face_enhance=true" \
  -F "face_upscale=3" \
  -o result.png
```

#### GET `/api/health`
Проверка работоспособности API.

**Ответ:** `{"status": "ok", "message": "API is running"}`

#### GET `/api/docs`
Автоматическая документация API (Swagger UI).

#### GET `/api/redoc`
Альтернативная документация API (ReDoc).

#### GET `/api/openapi.json`
OpenAPI спецификация в формате JSON.

## Документация API

Подробная спецификация API с примерами для разных языков программирования доступна в файле [src/API.md](src/API.md).

#### GET `/api/docs`
Автоматическая документация API (Swagger UI).

### CLI режим

```bash
# Базовый запуск
python src/main.py src/image.jpg src/ref.png -o result.png

# С улучшением лица и фоном
python src/main.py src/image.jpg src/ref.png -o result.png \
  --face-enhance --face-upscale 3 \
  --background src/bg.jpg

# Использование пресетов
python src/main.py src/image.jpg src/ref.png -o result.png --preset face3
```

## Параметры командной строки

| Параметр | Описание |
|----------|----------|
| `portrait` | Путь к исходному портрету |
| `reference` | Путь к изображению-референсу по цвету |
| `-o, --output` | Путь для сохранения результата (по умолчанию: `result.png`) |
| `--face-enhance` | Включить улучшение лица через GFPGAN |
| `--face-upscale` | Масштаб для GFPGAN (1-4, рекомендуется 2-4, выше = лучше качество) |
| `--face-strength` | Интенсивность улучшения (0.0-1.0, где 1.0 = полный эффект, по умолчанию 1.0) |
| `--face-iterations` | Количество итераций улучшения (1-3, больше = сильнее эффект, по умолчанию 1) |
| `--background` | Путь к фону (будет подложен под результат) |
| `--bg-model` | Модель удаления фона: isnet-general-use (лучшая), u2net_human_seg, u2net, silueta, u2netp |
| `--alpha-erode` | Сжать маску на N пикселей (убирает ореолы) |
| `--alpha-dilate` | Расширить маску на N пикселей (после эрозии) |
| `--alpha-feather` | Размытие края маски (Gaussian blur, пиксели) |
| `--keep-all` | Оставить все объекты на маске (не только главного человека). По умолчанию оставляется только самый большой объект. |
| `--preset {face3,face8}` | Готовые наборы настроек |

## Модели удаления фона

Доступны следующие модели (выбираются через параметр `--bg-model` или `bg_model` в API):

- **`isnet-general-use`** ⭐ (по умолчанию) — ISNet, лучшая точность для портретов (~200 МБ)
- **`u2net_human_seg`** — U2Net для людей, хорошо для портретов, быстрее ISNet (~176 МБ)
- **`u2net`** — U2Net базовая, универсальная и быстрая (~176 МБ)
- **`silueta`** — Silueta, хорошая для общих случаев (~43 МБ)
- **`u2netp`** — U2Net Lite, самая быстрая, но менее точная (~4 МБ)

### Установка моделей

**Автоматическая установка (рекомендуется):**
Модели скачиваются автоматически при первом использовании. Просто запустите обработку с нужной моделью:

```bash
# CLI
python src/main.py src/image.jpg src/ref.png -o result.png --bg-model isnet-general-use

# Или через веб-интерфейс - просто выберите модель в dropdown
```

**Ручная установка (предзагрузка):**
Если хотите предзагрузить все модели заранее, используйте готовый скрипт:

```bash
python src/download_models.py
```

Этот скрипт скачает все доступные модели и покажет прогресс.

**Расположение моделей:**
Модели сохраняются в:
- **Windows:** `C:\Users\<username>\.u2net\` или `C:\Users\<username>\.isnet\`
- **Linux/macOS:** `~/.u2net/` или `~/.isnet/`

**Проверка установленных моделей:**
```bash
# Windows PowerShell
Get-ChildItem $env:USERPROFILE\.u2net, $env:USERPROFILE\.isnet -ErrorAction SilentlyContinue

# Linux/macOS
ls ~/.u2net/ ~/.isnet/ 2>/dev/null
```

**Очистка кэша моделей:**
Если нужно удалить модели для освобождения места:
```bash
# Windows PowerShell
Remove-Item $env:USERPROFILE\.u2net -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item $env:USERPROFILE\.isnet -Recurse -Force -ErrorAction SilentlyContinue

# Linux/macOS
rm -rf ~/.u2net/ ~/.isnet/
```

## Требования

- Python 3.10+
- Зависимости из `src/requirements.txt`
- При первом запуске автоматически скачаются модели удаления фона и GFPGAN (~330 МБ)

## Решение проблем

### Ошибка активации виртуального окружения в PowerShell

Если при активации `.venv\Scripts\activate` возникает ошибка о политике выполнения скриптов:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Или используйте `activate.bat` вместо PowerShell скрипта:
```cmd
.venv\Scripts\activate.bat
```

## Лицензия

Проект для личного использования.
