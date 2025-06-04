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
    *   **EfficientNet-Lite4 (TFLite Image Classification)**: Performs image classification using the highly efficient EfficientNet-Lite4 model, running via the TensorFlow Lite runtime. This demo showcases optimized model performance for edge devices and includes experimental support for GPU delegation.
    *   **TinyBERT Sentiment**: Analyzes text sentiment (positive/negative) using a TinyBERT base model with a simple classification head.
    *   **Interactive LLM Demo**: Allows users to interact with a locally configured Large Language Model (LLM) via an OpenAI-compatible API.
    *   **Object Detection (NanoDet-Plus with NCNN)**: Detects common objects in images using the lightweight NanoDet-Plus model, executed by the NCNN inference engine. This demonstrates high-speed CPU-based object detection suitable for edge applications. (Note: Full NCNN post-processing is currently a placeholder in the Python integration).
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
    *Note: This step requires sufficient disk space, as libraries like PyTorch, Transformers, and TensorFlow can be large. This now includes `tensorflow-lite` for the EfficientNet demo, and `pyncnn` with `opencv-python-headless` for the NCNN object detection demo.*
5.  **(For LLM Demo Only) Set up Local LLM Server & Configure Endpoint**:
    *   The "Interactive LLM Demo" requires you to run a separate local LLM server (like `text-generation-webui` with Oobabooga, or Ollama with its OpenAI compatible API).
    *   Ensure your chosen LLM server is running and exposes an OpenAI-compatible API endpoint (e.g., `http://localhost:5000/v1` or `http://localhost:11434/v1`).
    *   Open the `webapp/llm_config.py` file.
    *   Modify the `LLM_API_ENDPOINT` variable to match the URL of your local LLM server.
6.  **(For EfficientNet-Lite4 Demo) Download TFLite Model**:
    *   The EfficientNet-Lite4 demo requires a TFLite model file. Due to potential download issues from direct links, you may need to manually download `efficientnet_lite4_classification_2.tflite` and place it in the `webapp/models/tflite/` directory. A placeholder file (`efficientnet_lite4_classification_2.tflite.PLEASE_DOWNLOAD_MANUALLY`) is created by default if the automatic download fails. You can search for "EfficientNet-Lite4 classification tflite" on TensorFlow Hub to find a download source.
7.  **(For NanoDet-Plus NCNN Demo) Setup NCNN and Model Files**:
    *   The NanoDet-Plus demo uses `pyncnn`, a Python wrapper for the NCNN library. `pyncnn` might require the NCNN C++ library to be built and installed on your system first, if `pip install pyncnn` encounters issues. Refer to official NCNN and `pyncnn` documentation for detailed NCNN installation.
    *   The NCNN model files (`nanodet-plus-m_320.param` and `nanodet-plus-m_320.bin`) are attempted to be downloaded automatically. If this fails, or you wish to use a different NCNN model, place your `.param` and `.bin` files in the `webapp/models/ncnn/` directory. Placeholder files (`*.PLEASE_DOWNLOAD_MANUALLY`) might be created if the automatic download fails.
8.  Navigate to the web application directory:
    ```bash
    cd webapp
    ```
9.  Run the Flask application:
    ```bash
    python app.py
    ```
    The application will start, and by default, it should be accessible at `http://127.0.0.1:5000` in your web browser.

10. Open your web browser and go to `http://127.0.0.1:5000`.

### Manual Testing for the Web Application

To ensure the web application is functioning correctly, perform the following manual checks:

*   **Homepage (`/`)**:
    *   Loads correctly with the title "Welcome - AI Model Gallery".
    *   Displays the welcome message and introduction to the gallery.
    *   Navigation links ("Home", "Model Gallery") are present and functional.
*   **Model Gallery Page (`/gallery`)**:
    *   Loads correctly when accessed via the navigation link.
    *   Displays the available models: Emotion Detector, EfficientNet-Lite4 (TFLite Image Classification), TinyBERT Sentiment, Interactive LLM Demo.
    *   Each model entry shows a name, description, and a "Try Demo" button.
    *   "Try Demo" buttons navigate to the correct demo pages.
*   **Emotion Detector Demo Page (`/demo/emotion-detector`)**:
    *   Loads with the title "Emotion Detector Demo - AI Model Gallery".
    *   The explanation text about the model and its usage is visible.
    *   Entering text (e.g., "I am happy today") and submitting performs inference.
    *   The results section displays the original input text, the predicted emotion, and a confidence score.
    *   Submitting empty input is handled (e.g., shows "No specific emotion detected or input was empty").
*   **EfficientNet-Lite4 Demo Page (`/demo/mobilenet`)**:
    *   Loads with a title like "EfficientNet-Lite4 Demo - AI Model Gallery" (or ensure the template reflects this if titles are dynamic - the template `demo_mobilenet.html` would need this title change).
    *   The explanation text about the model should be updated to describe EfficientNet-Lite4 and its use of TFLite. (This would be in `demo_mobilenet.html`, not directly testable here but noted).
    *   Uploading a valid image file performs inference using the TFLite model.
    *   The top 5 classification results are shown.
    *   Check server logs for messages related to TFLite model loading and whether the GPU delegate was used (e.g., "TFLite model loaded successfully... WITH GPU delegate" or "...using CPU").
    *   Submitting without a file or with an unsupported file type is handled (e.g., shows an error message).
*   **TinyBERT Sentiment Demo Page (`/demo/tinybert`)**:
    *   Loads with the title "TinyBERT Sentiment Analysis Demo - AI Model Gallery" (or similar, based on final implementation).
    *   The explanation text about the model is visible.
    *   Entering text (e.g., "This movie was fantastic!") and submitting performs inference.
    *   The results section displays the original input text, the predicted sentiment (positive/negative), and a confidence score.
*   **Interactive LLM Demo Page (`/demo/llm`)**:
    *   Loads with the title "LLM Demo".
    *   Displays the informational note about requiring a local LLM and the configured endpoint.
    *   Entering a prompt (e.g., "What is the capital of France?") and submitting performs a query to the configured LLM.
    *   The LLM's response is displayed in the response area.
    *   If the LLM server is not running or the endpoint is incorrect, an appropriate error message from the `llm_client` should be displayed.
    *   Test the blacklist: If your LLM is running and you can get it to say one of the words in `webapp/blacklist.txt` (e.g., "example_unsafe_word"), the output should be "[Content filtered due to potentially sensitive subject matter.]".
*   **Object Detection Demo Page (`/demo/object-detection`)**:
    *   Loads with the title "Object Detection Demo (NanoDet-Plus NCNN)".
    *   Displays informational notes about NCNN, model status, and placeholder post-processing.
    *   If NCNN or the model files are unavailable, an appropriate error message should be shown. Check server logs for `pyncnn` import status and NCNN model loading messages.
    *   Uploading an image file attempts inference.
    *   The original image and a version (intended for bounding boxes) are displayed. (Note: Due to placeholder post-processing in `ncnn_object_detector.py`, actual bounding boxes may not appear initially. The key is to check if the NCNN inference pipeline runs without crashing and the page displays the processed image, even if no boxes are drawn.)
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
- [Android Deployment Guide](docs/android_deployment.md)

## 🤝 Communauté

Rejoignez-nous sur les meetups IoT avec nos T-shirts "Mon IA tient dans 100 Ko" !

## 📄 License

Ce projet est sous licence Apache 2.0. Consultez le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

Merci à tous les contributeurs et à la communauté open source pour leur soutien !

Maintenu par @kabir308
