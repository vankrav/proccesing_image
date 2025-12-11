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

### Windows (PowerShell)

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

# 4. Запустить обработку
python src/main.py src/image.jpg src/ref.png -o result.png
```

### Linux/macOS

```bash
# 1. Создать виртуальное окружение
python3 -m venv .venv

# 2. Активировать
source .venv/bin/activate

# 3. Установить зависимости
pip install -r src/requirements.txt

# 4. Запустить обработку
python src/main.py src/image.jpg src/ref.png -o result.png
```

## Основные возможности

- **Удаление фона** — автоматическое удаление фона с помощью rembg/u2net
- **Перенос цвета** — приведение портрета к цветам референса (метод Рейнхарда в Lab)
- **Улучшение лица** — сглаживание кожи через GFPGAN (опционально)
- **Очистка краев** — эрозия, дилятация и размытие для чистых краев
- **Наложение фона** — подкладывание заданного фона под результат

## Примеры использования

### Базовый запуск
```bash
python src/main.py src/image.jpg src/ref.png -o result.png
```

### С улучшением лица и фоном
```bash
python src/main.py src/image.jpg src/ref.png -o result.png \
  --face-enhance --face-upscale 3 \
  --background src/bg.jpg
```

### Использование пресетов
```bash
# Пресет face3 (upscale=3, фон bg.jpg)
python src/main.py src/image.jpg src/ref.png -o result.png --preset face3

# Пресет face8 (upscale=8, фон bg.jpg)
python src/main.py src/image.jpg src/ref.png -o result.png --preset face8
```

### Полная настройка
```bash
python src/main.py src/image.jpg src/ref.png -o result.png \
  --face-enhance --face-upscale 2 \
  --alpha-erode 2 --alpha-dilate 2 --alpha-feather 5 \
  --background src/bg.jpg
```

## Параметры командной строки

| Параметр | Описание |
|----------|----------|
| `portrait` | Путь к исходному портрету |
| `reference` | Путь к изображению-референсу по цвету |
| `-o, --output` | Путь для сохранения результата (по умолчанию: `result.png`) |
| `--face-enhance` | Включить улучшение лица через GFPGAN |
| `--face-upscale` | Масштаб для GFPGAN (1-4, рекомендуется 3) |
| `--background` | Путь к фону (будет подложен под результат) |
| `--alpha-erode` | Сжать маску на N пикселей (убирает ореолы) |
| `--alpha-dilate` | Расширить маску на N пикселей (после эрозии) |
| `--alpha-feather` | Размытие края маски (Gaussian blur, пиксели) |
| `--preset {face3,face8}` | Готовые наборы настроек |

## Требования

- Python 3.10+
- Зависимости из `src/requirements.txt`
- При первом запуске автоматически скачаются модели u2net и GFPGAN (~330 МБ)

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
