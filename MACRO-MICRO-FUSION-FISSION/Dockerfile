# Base image Alpine Linux
FROM alpine:latest

# Mettre à jour et installer les dépendances minimales
RUN apk add --no-cache \
    python3 \
    py3-pip \
    build-base \
    cmake \
    git \
    wget \
    curl \
    bash

# Installer les dépendances Python minimales
RUN pip3 install --no-cache-dir \
    torch==2.0.0 \
    transformers==4.30.0 \
    numpy==1.24.0 \
    onnxruntime==1.14.0

# Créer le répertoire de travail
WORKDIR /app

# Copier les fichiers nécessaires
COPY requirements.txt ./
COPY src/ ./src/
COPY models/ ./models/
COPY wasm/ ./wasm/

# Installer les dépendances spécifiques
RUN pip3 install --no-cache-dir -r requirements.txt

# Compiler les modèles en WebAssembly
RUN cd wasm && make

# Commande par défaut
CMD ["python3", "src/main.py"]
