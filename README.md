# Portrait Processing CLI

# Что делает
CLI-скрипт для портретов:
- удаляет фон (rembg / u2net);
- подтягивает цвета под референс (Reinhard в Lab с учётом маски);
- опционально улучшает лицо (GFPGAN);
- чистит края маски (эрозия/дилятация/feather);
- подкладывает заданный фон.

## Требования
- Python 3.10+ (тестировалось в venv);
- Установленные зависимости: `pip install -r requirements.txt`.
- При первом запуске докачаются веса u2net и GFPGAN (~330 МБ).

## Быстрый старт
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Базовый прогон
python main.py image.jpg ref.png -o result.png
```

## Аргументы
- `portrait` — путь к портрету (входное изображение).
- `reference` — путь к референсу по цвету.
- `-o/--output` — файл результата (PNG/др).
- `--face-enhance` — включить GFPGAN.
- `--face-upscale` — масштаб GFPGAN (1–4; 8 допустим, но тяжёлый).
- `--background` — путь к фону, подложится под итог с альфой.
- `--alpha-erode` — сжать маску на N пикселей.
- `--alpha-dilate` — расширить маску на N после эрозии.
- `--alpha-feather` — размытие края (пиксели, Gaussian).
- `--preset {face3,face8}` — готовые наборы (см. ниже).

## Пресеты (готовые настройки)
- `face3` → `--face-enhance --face-upscale 3 --background bg.jpg`
- `face8` → `--face-enhance --face-upscale 8 --background bg.jpg`

Пример:
```bash
python main.py image.jpg ref.png -o result.png --preset face3
python main.py image.jpg ref.png -o result_face8.png --preset face8
```

## Примеры ручной настройки
- Мягкий край и фон:
```bash
python main.py image.jpg ref.png -o result.png \
  --face-enhance --face-upscale 2 \
  --alpha-erode 2 --alpha-dilate 2 --alpha-feather 5 \
  --background bg.jpg
```

- Без улучшения лица, только цвет + фон:
```bash
python main.py image.jpg ref.png -o result.png --background bg.jpg
```

## Файлы в репо
- `main.py` — вся логика пайплайна.
- `requirements.txt` — зависимости.
- `image.jpg`, `image2.jpg` — примеры входа.
- `ref.png` — пример референса по цвету.
- `bg.jpg` — пример фона.
- `result*.png` — примеры результатов.
- `gfpgan/weights/` — закешированные веса GFPGAN (скачаются при первом запуске).

## Типичные советы по качеству
- Края: подбирайте `--alpha-erode/--alpha-dilate/--alpha-feather`.
- Цвет: для сложных референсов иногда полезно обрезать фон на референсе заранее.
- GFPGAN: `--face-upscale 3–4` чаще всего оптимум; 8 — тяжелее, может выглядеть «пластиково».
## Частые проблемы
- `onnxruntime`/`torch` не найден: убедитесь, что активирован venv и зависимости установлены.
- Долгая загрузка при первом старте — качаются модели u2net и GFPGAN.
- Вырезка с ореолами — увеличьте `--alpha-erode`, затем верните объём `--alpha-dilate`, сгладьте `--alpha-feather`.
