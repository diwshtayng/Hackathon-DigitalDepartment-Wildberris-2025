# Hackathon-DigitalDepartment-Wildberris-2025
# Flask-приложение в Docker

Простое Flask-приложение, упакованное в Docker-контейнер.

## 📦 Требования
- Docker Desktop
- Python 3.9+

## 🚀 Быстрый старт

### 1. Клонируйте репозиторий
```bash
git clone https://github.com/diwshtayng/Hackathon-DigitalDepartment-Wildberris-2025.git
cd our-flask-app
```

### 2. Соберите и запустите контейнер
```bash
docker build -t our--flask-app .
docker run -dp 5002:5002 our-flask-app
```

### 3. Откройте в браузере
👉 [http://localhost:5002](http://localhost:5002)

## 🛠 Структура проекта
```
Hackathon-DigitalDepartment-Wildberris-2025/
├── app/
│   ├── __init__.py           # (опционально) инициализация Flask-приложения
│   ├── app.py                # Основной файл с Flask-приложением и маршрутами
│   ├── model.pkl             # Сериализованная обученная модель
│   ├── preprocessor.py       # Скрипт для предобработки входных данных перед предсказанием
│   ├── static/               # Место для CSS, JS, изображений, иконок и др.
│   └── templates/
│       └── index.html        # HTML-шаблон формы ввода данных и отображения результата
├── .dockerignore             # Файл с правилами исключения из Docker-контекста
├── Dockerfile                # Инструкция для сборки Docker-образа приложения
├── README.md                 # Документация к проекту: описание, как запускать, использовать и т.д.
└── requirements.txt          # Список зависимостей Python (Flask, pandas, scikit-learn и др.)
```
