import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Cargar el modelo y el vectorizador
loaded_model = joblib.load('linear_svc_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')  # Asegúrate de haber guardado el vectorizador también

# Ahora puedes usar loaded_model y vectorizer para hacer predicciones
new_texts = ["I love my life"]
new_texts_transformed = vectorizer.transform(new_texts)
predictions = loaded_model.predict(new_texts_transformed)

for text, prediction in zip(new_texts, predictions):
    print(f'Texto: {text} | Predicción: {prediction}')