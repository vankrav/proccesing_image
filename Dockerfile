# Используем официальный образ Python
FROM python:3.10-slim

# Устанавливаем системные зависимости для работы с изображениями и ML библиотеками
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файлы зависимостей
COPY src/requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь исходный код
COPY src/ ./src/
COPY config.json ./

# Создаем необходимые директории
RUN mkdir -p uploads results models/.u2net models/.isnet

# Устанавливаем переменные окружения для моделей
ENV U2NET_HOME=/app/models/.u2net
ENV ISNET_HOME=/app/models/.isnet
ENV HOME=/app

# Открываем порт для FastAPI
EXPOSE 8000

# Команда запуска приложения
CMD ["python", "src/app.py"]







