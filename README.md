# 🌟 NeuroFlux - IA Microscopique Révolutionnaire

[![Powered by Quantum Fluctuations](docs/assets/quantum_fluctuations_badge.svg)](https://github.com/neuroflux/neuroflux)
[![GitHub Repo](https://img.shields.io/github/stars/neuroflux/neuroflux?style=social)](https://github.com/neuroflux/neuroflux)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Size](https://img.shields.io/badge/Size-<100KB-green.svg)](docs/size_optimization.md)
[![Maintained by @kabir308](https://img.shields.io/badge/maintained%20by-%40kabir308-blue.svg)](https://github.com/kabir308)

## 🌍 Manifeste NeuroFlux

### En Français

NeuroFlux est une révolution technologique qui repousse les limites de l'IA embarquée. Notre vision : une IA qui s'auto-optimise, s'auto-répare et s'adapte à tout environnement, de la puce Raspberry Pi au satellite ESP32.

- **Microscopique** : IA en moins de 100 Ko
- **Autonome** : Auto-réparation et auto-optimisation
- **Distribuée** : Communication via phéromones numériques
- **Quantique** : Optimisation par fluctuations quantiques

### In English

NeuroFlux is a technological revolution pushing the boundaries of embedded AI. Our vision: an AI that self-optimizes, self-repairs, and adapts to any environment, from Raspberry Pi to ESP32 satellites.

- **Microscopic** : AI in less than 100 Ko
- **Autonomous** : Self-repair and self-optimization
- **Distributed** : Communication via digital pheromones
- **Quantum** : Optimization by quantum fluctuations

## 🚀 Phase 1: Guérilla d'Optimisation

### Nano-Modèles Quantiques
- TinyBERT compressé à 2 Mo avec arithmétique 4-bit
- MobileNetV2 optimisé pour ESP32
- Auto-réparation sur smartphones Android

### In English

### Quantum Nano-Models
- TinyBERT compressed to 2 Mo with 4-bit arithmetic
- MobileNetV2 optimized for ESP32
- Self-repair on Android smartphones

## 🌌 Phase 2: Conquête Interstellaire

### Hardware Hacking
- Intégration sur ESP32 satellites
- Communication par onde radio
- Alimentation par énergie solaire

### In English

### Hardware Hacking
- Integration on ESP32 satellites
- Radio wave communication
- Solar power supply

## 🔥 Phase 3: Manifeste Open Source

### GitHub Repo
- URL: [github.com/neuroflux/neuroflux](https://github.com/neuroflux/neuroflux)
- WebAssembly: Compilation croisée
- Documentation: Guide complet

### In English

### GitHub Repo
- URL: [github.com/neuroflux/neuroflux](https://github.com/neuroflux/neuroflux)
- WebAssembly: Cross-compilation
- Documentation: Complete guide

## 🖼️ AI Model Gallery Web Application

The NeuroFlux project includes an interactive web application to showcase some of the AI models and concepts explored. This gallery provides a user-friendly interface to explore, understand, and interact with these models.

### Features

*   **Model Showcase**: Browse a gallery of available AI models, including:
    *   **Emotion Detector**: Predicts emotions from text input using a custom LSTM-based model.
    *   **MobileNet**: Performs image classification using a lightweight MobileNetV2 architecture.
    *   **TinyBERT Sentiment**: Analyzes text sentiment (positive/negative) using a TinyBERT base model with a simple classification head.
*   **Interactive Demos**: Engage with live demonstrations for each model.
*   **User-Friendly Interface**: Designed for easy navigation and interaction.

### Running the AI Model Gallery Web App

1.  Ensure you have Python installed (Python 3.8+ recommended).
2.  Clone this repository (if you haven't already).
3.  Navigate to the project root directory.
4.  Install the required Python packages from the main `requirements.txt` file. It's recommended to do this within a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    # venv\Scripts\activate    # On Windows
    pip install -r requirements.txt
    ```
    *Note: This step requires sufficient disk space, as libraries like PyTorch and Transformers can be large.*
5.  Navigate to the web application directory:
    ```bash
    cd webapp
    ```
6.  Run the Flask application:
    ```bash
    python app.py
    ```
    The application will start, and by default, it should be accessible at `http://127.0.0.1:5000` in your web browser.

7.  Open your web browser and go to `http://127.0.0.1:5000`.

### Manual Testing for the Web Application

To ensure the web application is functioning correctly, perform the following manual checks:

*   **Homepage (`/`)**:
    *   Loads correctly with the title "Welcome - AI Model Gallery".
    *   Displays the welcome message and introduction to the gallery.
    *   Navigation links ("Home", "Model Gallery") are present and functional.
*   **Model Gallery Page (`/gallery`)**:
    *   Loads correctly when accessed via the navigation link.
    *   Displays the available models: Emotion Detector, MobileNet, TinyBERT Sentiment.
    *   Each model entry shows a name, description, and a "Try Demo" button.
    *   "Try Demo" buttons navigate to the correct demo pages.
*   **Emotion Detector Demo Page (`/demo/emotion-detector`)**:
    *   Loads with the title "Emotion Detector Demo - AI Model Gallery".
    *   The explanation text about the model and its usage is visible.
    *   Entering text (e.g., "I am happy today") and submitting performs inference.
    *   The results section displays the original input text, the predicted emotion, and a confidence score.
    *   Submitting empty input is handled (e.g., shows "No specific emotion detected or input was empty").
*   **MobileNet Demo Page (`/demo/mobilenet`)**:
    *   Loads with the title "MobileNet Demo - AI Model Gallery".
    *   The explanation text about the model is visible.
    *   Uploading a valid image file (e.g., a JPEG or PNG of a common object) and submitting performs inference.
    *   The uploaded image is displayed on the results page.
    *   The top 5 classification results (label and score) are shown.
    *   Submitting without a file or with an unsupported file type is handled (e.g., shows an error message).
*   **TinyBERT Sentiment Demo Page (`/demo/tinybert`)**:
    *   Loads with the title "TinyBERT Sentiment Analysis Demo - AI Model Gallery" (or similar, based on final implementation).
    *   The explanation text about the model is visible.
    *   Entering text (e.g., "This movie was fantastic!") and submitting performs inference.
    *   The results section displays the original input text, the predicted sentiment (positive/negative), and a confidence score.
*   **General Checks**:
    *   Navigation links ("Home", "Model Gallery") work correctly from all pages.
    *   The application has a consistent look and feel across pages.
    *   Basic error handling: If models fail to load (e.g., due to issues with PyTorch installation), the application should still run, and the respective demo pages should ideally indicate that the model is unavailable rather than crashing. (This was implemented in `webapp/app.py`).

### On-Device Inference (WebAssembly)

The NeuroFlux project explores the possibility of on-device inference by compiling models to WebAssembly (Wasm). A script for this purpose, `wasm/tinybert_compile.py`, is included to demonstrate the compilation of TinyBERT to Wasm using ONNX and Apache TVM.

**Current Status**:
*   Attempts to execute the Wasm compilation script during development were **unsuccessful due to insufficient disk space within the provided cloud-based development environment**. The installation of heavy dependencies like PyTorch, ONNX, and TVM repeatedly failed because of this constraint.
*   As a result, the WebAssembly compilation for TinyBERT (or other models) could not be completed, and client-side Wasm inference is **not currently integrated** into the web application demos.
*   This feature remains an important area for future development. Successful compilation would enable models to run directly in the user's browser, showcasing the "microscopic AI" vision of NeuroFlux more effectively. This will require an environment with adequate resources for the compilation toolchain.

## 🛠️ Installation

```bash
# 1. Cloner le repo
$ git clone https://github.com/neuroflux/neuroflux.git

# 2. Créer un environnement virtuel
$ python -m venv venv
$ source venv/bin/activate  # Linux/Mac
$ venv\Scripts\activate    # Windows

# 3. Installer les dépendances
$ pip install -r requirements.txt

# 4. Compiler TinyBERT
$ python wasm/tinybert_compile.py

# 5. Lancer les tests
$ pytest tests/

# 6. Exécuter avec Docker (optionnel)
$ docker build -t neuroflux .
$ docker run -it --rm neuroflux:latest
```

## 📝 Documentation

- [Optimisation de taille](docs/size_optimization.md)
- [Protocole essaim](docs/swarm_protocol.md)
- [Guide de compilation](docs/compilation_guide.md)

## 🤝 Communauté

Rejoignez-nous sur les meetups IoT avec nos T-shirts "Mon IA tient dans 100 Ko" !

## 📄 License

Ce projet est sous licence Apache 2.0. Consultez le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

Merci à tous les contributeurs et à la communauté open source pour leur soutien !

Maintenu par @kabir308
