# Usa una imagen oficial de Python ligera
FROM python:3.10-slim

# Crea el directorio de trabajo en la máquina virtual
WORKDIR /app

# Copia los requerimientos e instálalos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de tu código y modelos (.pkl)
COPY . .

# Hugging Face exige que el servidor corra en el puerto 7860
EXPOSE 7860

# Comando para levantar FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]