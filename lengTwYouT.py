import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report

# Leer los datos y reducir el conjunto de datos
columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
df = pd.read_csv('../training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
#en caso de querer limitar la prueba
#df = df.sample(n=100000, random_state=42)

# Mapear las etiquetas a 'negative' y 'positive'
df[0] = df[0].replace({0: 'negative', 4: 'positive'})

x = df[5]
y = df[0]

# Crear el vectorizador y transformar los datos
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

# Dividir el conjunto de datos
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.01, random_state=42)

# Entrenar el modelo
#clf= MLPClassifier(activation='logistic', hidden_layer_sizes=(10,),solver='sgd')
model = LinearSVC()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred,target_names=['negative','positive']))

# Hacer predicciones para una nueva oración
new_text = ["I would like them to change the way they talk to us"]

# Transformar la nueva oración con el mismo vectorizador
new_text_vectorized = vectorizer.transform(new_text)

# Hacer la predicción
prediction = model.predict(new_text_vectorized)
print(f'Texto: {new_text[0]} | Predicción: {prediction[0]}')


# esto es para guardar el modelo
import joblib

# Guardar el modelo
joblib.dump(model, 'linear_svc_model.joblib')
# Suponiendo que 'vectorizer' es tu objeto TfidfVectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# Cargar el modelo
loaded_model = joblib.load('linear_svc_model.joblib')

# Ahora puedes usar loaded_model para hacer predicciones, por ejemplo:
new_texts = ["I hate my life", "I love machine learning"]
new_texts_transformed = vectorizer.transform(new_texts)
predictions = loaded_model.predict(new_texts_transformed)

for text, prediction in zip(new_texts, predictions):
    print(f'Texto: {text} | Predicción: {prediction}')
