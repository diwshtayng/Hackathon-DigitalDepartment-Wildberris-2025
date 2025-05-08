from flask import Flask, render_template, request, jsonify
from preprocessor import preprocess
import pandas as pd
import pickle
import os
import sklearn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Ошибка при загрузке модели: {e}")
    model = None


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_file", methods=["POST"])
def predict_file():
    # Проверка: есть ли файл
    if "file" not in request.files:
        return jsonify({"error": "Файл не передан"}), 400

    file = request.files["file"]
    
    try:
        # Читаем CSV-файл в DataFrame
        df = pd.read_csv(file)
        df = preprocess(df)

        # При необходимости — вызвать трансформации:
        # df = preprocess(df)  ← если ты используешь свой preprocessor.py

        # Предсказание
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })
    
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки файла: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

