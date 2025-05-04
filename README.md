# Hackathon-DigitalDepartment-Wildberris-2025
# Flask-приложение в Docker

Простое Flask-приложение, упакованное в Docker-контейнер.

## 📦 Требования
- Docker Desktop
- Python 3.9+

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/ваш-username/our-flask-app.git
cd our-flask-app
```

### 2. Соберите и запустите контейнер
```bash
docker build -t my-flask-app .
docker run -dp 5002:5002 my-flask-app
```

### 3. Откройте в браузере
👉 [http://localhost:5002](http://localhost:5002)

## 🛠 Структура проекта
```
our-flask-app/
├── app.py              # Основной файл Flask
├── requirements.txt    # Зависимости Python
├── Dockerfile          # Конфигурация Docker
└── README.md           # Этот файл
```
