from flask import Flask, render_template, request, jsonify
from preprocessor import preprocess
import pandas as pd
import pickle
import os
import sklearn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_correct.pkl")

try:
    with open('/Users/sergegribo/Desktop/ML/git_hackaton/Hackathon-DigitalDepartment-Wildberris-2025/our-flask-app/catboost_correct.pkl', "rb") as f:
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
        df = pd.read_csv(file)
        df['CreatedDate'] = pd.to_datetime(df['CreatedDate'])
        df['order_hour'] = df['CreatedDate'].dt.hour
        df.drop('CreatedDate', axis=1, inplace=True)
        df.drop(columns=['nm_id', 'user_id'], inplace=True)
        
    except Exception as e:
        return jsonify({"error": f"Ошибка обработки файла: {str(e)}"}), 500

    try:
        prediction = model.predict(df)
        probability = model.predict_proba(df)[:, 1]

        return jsonify({
            "prediction": prediction,
            "probability": probability
        })
    
    except Exception as er:
        return jsonify({"error": f"Ошибка в работе модели: {str(er)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)

