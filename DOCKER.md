# Запуск проекта через Docker

## Быстрый старт

### 1. Запуск через Docker Compose (рекомендуется)

```bash
# Запустить проект
docker-compose up -d

# Просмотр логов
docker-compose logs -f

# Остановить проект
docker-compose down
```

После запуска приложение будет доступно по адресу: **http://localhost:8000**

### 2. Запуск через Docker напрямую

```bash
# Сборка образа
docker build -t portrait-processing .

# Запуск контейнера
docker run -d \
  --name portrait-processing-app \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/src:/app/src \
  --restart unless-stopped \
  portrait-processing
```

## Структура volumes

- `./models` - кэш моделей машинного обучения (скачиваются автоматически при первом использовании)
- `./uploads` - временные файлы загрузок
- `./results` - результаты обработки
- `./config.json` - конфигурационный файл с настройками
- `./src` - исходный код (для доступа к ref.png и bg.jpg)

## Полезные команды

```bash
# Пересобрать образ после изменений
docker-compose build --no-cache

# Перезапустить контейнер
docker-compose restart

# Просмотр статуса
docker-compose ps

# Войти в контейнер
docker-compose exec portrait-processing bash

# Очистить volumes (удалить модели и результаты)
docker-compose down -v
```

## Проверка работоспособности

После запуска проверьте, что API работает:

```bash
curl http://localhost:8000/api/health
```

Должен вернуться ответ: `{"status":"ok","message":"API is running"}`

## Решение проблем

### Порт 8000 уже занят

Измените порт в `docker-compose.yml`:

```yaml
ports:
  - "8080:8000"  # Используйте другой порт вместо 8000
```

### Модели не скачиваются

Модели скачиваются автоматически при первом использовании. Если возникают проблемы:

1. Проверьте доступность интернета в контейнере
2. Проверьте логи: `docker-compose logs -f`
3. Модели сохраняются в `./models` директории

### Недостаточно памяти

Обработка изображений требует достаточно памяти. Убедитесь, что у Docker выделено достаточно ресурсов:
- Минимум 4GB RAM
- Рекомендуется 8GB+ для комфортной работы



