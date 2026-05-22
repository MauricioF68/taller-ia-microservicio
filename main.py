import os
import re
import nltk
import joblib
import numpy as np
import unicodedata
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

app = FastAPI(
    title="Motor de IA - Producción",
    description="Microservicio NLP con extracción forzada de probabilidades y normalización estricta."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class CommandRequest(BaseModel):
    text: str

stop_words_es = set(stopwords.words('spanish'))
stop_words_es.update([
    'necesito', 'voy', 'tengo', 'hacer', 'urge', 'mantenimiento',
    'cliente', 'pide', 'solicito', 'requiero', 'usando', 'apoyo',
    'utilizando', 'alcancen', 'urgente', 'tráeme', 'pásame', 'usar',
    'iniciando', 'reparación', 'zona', 'tarea', 'dónde', 'está',
    'trabajo', 'empezar', 'requiere', 'contenedor', 'favor', 'búscame', 'hoy',
    'ayúdame', 'recomiéndame', 'algo', 'intento', 'intentando', 'dame', 'trae'
])

def limpiar_texto(texto):
    texto = str(texto).lower()
    
    # Remoción estricta de tildes
    texto = unicodedata.normalize('NFD', texto)
    texto = "".join([c for c in texto if unicodedata.category(c) != 'Mn'])
    
    texto = re.sub(r'[^\w\s]', '', texto)
    tokens = texto.split()
    
    tokens_limpios = [w for w in tokens if w not in stop_words_es and len(w) > 1]
    return " ".join(tokens_limpios)

try:
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    mlb = joblib.load('binarizador_mlb.pkl')
    modelo_multilabel = joblib.load('modelo_multilabel_rf.pkl')
    print("📦 Artefactos de ML cargados exitosamente.")
except Exception as e:
    print(f"❌ Error crítico al cargar los archivos (.pkl): {e}")

@app.post("/predict")
async def predict_tool(request: CommandRequest):
    texto_procesado = limpiar_texto(request.text)
    
    if not texto_procesado:
        return {
            "text_received": request.text,
            "text_cleaned": "",
            "predicted_labels": [],
            "predicted_label": "",
            "confidence": 0.0
        }

    X_matriz = tfidf.transform([texto_procesado])
    
    Y_pred = modelo_multilabel.predict(X_matriz)
    etiquetas_tupla = mlb.inverse_transform(Y_pred)
    lista_etiquetas = list(etiquetas_tupla[0]) if etiquetas_tupla else []
    
    ml_label_resultado = ""
    confianza_porcentaje = 0.0

    if len(lista_etiquetas) > 0:
        ml_label_resultado = lista_etiquetas[0]
        confianza_porcentaje = 100.0
        
    else:
        lista_probabilidades = modelo_multilabel.predict_proba(X_matriz)
        probs_positivas = [proba[0][1] if proba.shape[1] > 1 else 0.0 for proba in lista_probabilidades]
        
        indice_ganador = np.argmax(probs_positivas)
        probabilidad_ganadora = probs_positivas[indice_ganador]
        
        if probabilidad_ganadora > 0:
            ml_label_resultado = mlb.classes_[indice_ganador]
            confianza_porcentaje = round(float(probabilidad_ganadora) * 100, 2)

    return {
        "text_received": request.text,
        "text_cleaned": texto_procesado,
        "predicted_labels": lista_etiquetas,
        "predicted_label": ml_label_resultado,
        "confidence": confianza_porcentaje
    }