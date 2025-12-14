# Настройка GPU поддержки в Docker

## Требования

Для использования GPU в Docker контейнере необходимо:

1. **NVIDIA GPU** с поддержкой CUDA
2. **NVIDIA драйверы** (версия >= 450.80.02)
3. **NVIDIA Container Toolkit** (ранее nvidia-docker2)

## Установка NVIDIA Container Toolkit

### Windows (WSL2)

Если вы используете Docker Desktop на Windows с WSL2:

1. Установите NVIDIA драйверы для Windows
2. Установите WSL2 с Ubuntu
3. В WSL2 Ubuntu выполните:

```bash
# Добавляем репозиторий NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Устанавливаем nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапускаем Docker daemon
sudo systemctl restart docker
```

### Linux (Ubuntu/Debian)

```bash
# Добавляем репозиторий NVIDIA
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Устанавливаем nvidia-container-toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Перезапускаем Docker daemon
sudo systemctl restart docker
```

### Проверка установки

```bash
# Проверяем, что GPU доступен
nvidia-smi

# Проверяем, что Docker видит GPU
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Запуск с GPU поддержкой

### Вариант 1: Использование docker-compose.gpu.yml (рекомендуется)

```bash
# Запуск с GPU
docker-compose -f docker-compose.gpu.yml up -d --build

# Просмотр логов
docker-compose -f docker-compose.gpu.yml logs -f
```

### Вариант 2: Раскомментировать GPU в docker-compose.yml

Откройте `docker-compose.yml` и раскомментируйте секцию `deploy` в конце файла:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Затем запустите:

```bash
docker-compose up -d --build
```

### Вариант 3: Использование Docker напрямую

```bash
# Сборка образа
docker build -f Dockerfile.gpu -t portrait-processing-gpu .

# Запуск контейнера с GPU
docker run -d \
  --name portrait-processing-app-gpu \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/config.json:/app/config.json \
  -v $(pwd)/src:/app/src \
  --restart unless-stopped \
  portrait-processing-gpu
```

## Проверка работы GPU

После запуска контейнера проверьте использование GPU:

```bash
# Войти в контейнер
docker exec -it portrait-processing-app-gpu bash

# Проверить доступность GPU
nvidia-smi

# Проверить PyTorch
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

## Преимущества GPU версии

- **Ускорение обработки**: GFPGAN и модели удаления фона работают значительно быстрее на GPU
- **Лучшая производительность**: Обработка изображений может быть в 10-50 раз быстрее
- **Поддержка больших изображений**: GPU позволяет обрабатывать изображения большего разрешения

## Отличия CPU и GPU версий

| Характеристика | CPU версия | GPU версия |
|----------------|------------|------------|
| Базовый образ | python:3.10-slim | nvidia/cuda:12.1.0-runtime |
| Размер образа | ~1.5 GB | ~3-4 GB |
| PyTorch | CPU версия | CUDA версия |
| ONNX Runtime | CPU | GPU |
| Скорость обработки | Медленнее | Быстрее (10-50x) |
| Требования | Любой компьютер | NVIDIA GPU |

## Решение проблем

### GPU не определяется

1. Проверьте установку драйверов: `nvidia-smi`
2. Проверьте установку nvidia-container-toolkit
3. Перезапустите Docker daemon
4. Проверьте логи: `docker-compose logs -f`

### Ошибка "nvidia-container-runtime not found"

Установите nvidia-container-toolkit (см. инструкции выше).

### Ошибка "CUDA out of memory"

Уменьшите размер обрабатываемых изображений или используйте CPU версию для небольших изображений.







