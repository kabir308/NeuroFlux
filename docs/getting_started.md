# Getting Started with NeuroFlux

This guide provides the essential steps to get the NeuroFlux project up and running on your local machine, including prerequisites, cloning, and installation.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

*   **Python:**
    *   Version 3.8 or higher is required. Python 3.10 or later is recommended for optimal compatibility with all features.
    *   You can download Python from [python.org](https://www.python.org/downloads/).
    *   Ensure `python` and `pip` (Python's package installer) are in your system's PATH.

*   **Git:**
    *   Required for cloning the project repository.
    *   You can download Git from [git-scm.com](https://git-scm.com/downloads).

*   **Virtual Environment Tool (Recommended):**
    *   Python's built-in `venv` module is recommended for creating isolated project environments.
    *   Most Python installations include `venv` by default.
    *   Alternatively, you can use tools like `conda`.

*   **Operating System:**
    *   The project is primarily developed and tested on Linux (Ubuntu).
    *   It should also work on macOS and Windows (using Windows Subsystem for Linux - WSL - is recommended for the best experience on Windows, especially for some dependencies).

*   **Build Tools (for some advanced dependencies):**
    *   Some dependencies included in `requirements.txt` (like `pyncnn`, `tvm`, or certain components of `qiskit` or `pennylane`) may require C/C++ compilers (e.g., GCC, Clang, MSVC) and related build tools (e.g., CMake) to be installed on your system if they need to be compiled from source during installation.
    *   For specific libraries like NCNN (used by `pyncnn`), you might need to install the NCNN SDK separately. Refer to the respective library's documentation for detailed system requirements.
    *   These are typically not needed for basic web application usage or core Python functionalities but become relevant for advanced model compilation or hardware-specific features.
    <!-- TODO: In a future update, explicitly link ESP-IDF toolchain requirements to 'embedded/esp32/' and Emscripten/WASI SDK to 'wasm/' documentation if those sections are detailed. -->

*   **Docker (Optional but Recommended for Web Application):**
    *   If you plan to run the AI Model Gallery web application using Docker (see [Web Application Guide](./webapp_gallery_guide.md) and [Docker Deployment Guide](./deployment_webapp_docker.md)), you will need Docker installed.
    *   Download Docker from [docker.com](https://www.docker.com/products/docker-desktop).

*   **Sufficient Disk Space:**
    *   Installing all dependencies, especially those for deep learning and scientific computing (PyTorch, TensorFlow, Transformers, etc.), can consume several gigabytes of disk space. Ensure you have adequate free space.

## Installation

Once you have all the prerequisites, follow these steps to install NeuroFlux:

1.  **Clone the Repository:**
    Open your terminal or command prompt and clone the NeuroFlux repository from GitHub:
    ```bash
    git clone https://github.com/neuroflux/neuroflux.git
    cd neuroflux
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with your global Python packages.

    *   **Using `venv` (Python's built-in tool):**
        ```bash
        # Navigate to the cloned project root directory (if not already there)
        # cd neuroflux
        <!-- TODO: Consider adding a note to explicitly confirm the user is in the 'neuroflux' root directory before running 'python -m venv venv'. -->

        # Create a virtual environment (e.g., named 'venv')
        python -m venv venv

        # Activate the virtual environment:
        # On Linux/macOS:
        source venv/bin/activate
        # On Windows (Command Prompt/PowerShell):
        # venv\Scripts\activate
        ```
        After activation, your terminal prompt should change to indicate you are in the virtual environment (e.g., `(venv) your-prompt$`).

    *   **Using `conda` (Alternative):**
        If you prefer using Conda:
        ```bash
        conda create -n neuroflux_env python=3.10 # Or your preferred Python 3.8+ version
        conda activate neuroflux_env
        ```

3.  **Install Core Dependencies:**
    With your virtual environment activated, install the main project dependencies listed in `requirements.txt`. These are necessary for the core functionalities and the web application demo.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This step can take some time and requires a stable internet connection, as it will download and install several large libraries (e.g., PyTorch, Transformers, TensorFlow Lite).*

4.  **Install Hugging Face Specific Dependencies (Optional):**
    If you plan to work with scripts in the `huggingface/` directory (e.g., for using `NeuroFluxDataset`, running Gradio demos, or specific model training scripts located there), you'll need to install additional dependencies:
    ```bash
    pip install -r huggingface/requirements.txt
    ```

5.  **Verify Installation (Optional First Check):**
    You can perform a quick check to see if some basic components are recognized. For example, list the discoverable training scripts for existing models:
    ```bash
    python scripts/train_existing_models.py --list
    ```
    This command should list models like `emotion-detector`, `mobilenet`, etc., if the installation was successful and paths are correct.

You should now have NeuroFlux installed and ready for further exploration!

## Basic Project Structure Overview

After cloning and setting up the environment, you'll find the following key directories in the NeuroFlux project root:

*   `README.md`: The main readme file with a project overview and web app guide.
*   `docs/`: Contains all project documentation, including this guide. (See `docs/index.md` for a full list).
*   `src/`: Core Python source code for NeuroFlux, including modules for:
    *   `autopoietic/`: Self-repair and autopoietic systems.
    *   `pheromones/`: Digital pheromone communication.
    *   `security/`: Advanced security mechanisms.
    *   `advanced_systems/`: Bio-mimetic and other advanced concepts.
    *   `models/`: Some base model definitions.
    *   `engines/`: Orchestration engines (less developed).
*   `huggingface/`: Scripts and configurations related to Hugging Face models, including model definitions, training scripts, and the `NeuroFluxDataset` utility for TFLite conversion.
*   `webapp/`: Source code for the AI Model Gallery Flask web application.
*   `scripts/`: Utility and master training scripts (e.g., `train_existing_models.py`, `train_conceptual_systems.py`).
*   `embedded/esp32/`: C++ code for ESP32 microcontroller integration.
*   `wasm/`: Scripts and notes related to WebAssembly compilation of models.
*   `tests/`: Project automated tests.
*   `requirements.txt`: Core Python dependencies.
*   `huggingface/requirements.txt`: Dependencies for Hugging Face related scripts.
*   `Dockerfile`: For building a Docker image of the web application.

For a more detailed explanation of the project architecture, please refer to the [Project Architecture document](./architecture.md).

## Quick Verification & First Exploration (Post-Installation)

After following the installation steps, here are a few commands you can run to verify your setup and start exploring NeuroFlux:

1.  **List Trainable Existing Models:**
    This command, which you might have run as a final installation check, ensures the project paths are okay and lists models that have dedicated training scripts.
    ```bash
    python scripts/train_existing_models.py --list
    ```

2.  **Run Project Tests (Recommended):**
    Executing the automated tests is a good way to ensure that the core functionalities are working as expected in your environment.
    ```bash
    pytest tests/ -v
    ```
    *(Note: Some tests might require specific data or configurations. If tests fail, please check their specific error messages or consult the [Testing Guide](./testing_guide.md) for more details.)*

3.  **Explore the AI Model Gallery Web Application:**
    The web application provides interactive demos of several models.
    ```bash
    cd webapp
    python app.py
    ```
    Then open your browser to `http://127.0.0.1:5000`. (Refer to the [AI Model Gallery Web App Guide](./webapp_gallery_guide.md) for detailed setup if you encounter issues, especially with model downloads or specific demos like the LLM or NCNN demos).

4.  **Simulate Training for a Conceptual System:**
    You can explore the framework for training advanced conceptual systems:
    ```bash
    python scripts/train_conceptual_systems.py --list
    # then, for example:
    python scripts/train_conceptual_systems.py predictive_failure --epochs 1
    ```
    This will print the simulated workflow for training the selected conceptual model.

These steps should help you confirm that NeuroFlux is set up correctly and give you a starting point for exploring its capabilities.

## Next Steps

Congratulations on getting NeuroFlux set up! Now that you have a working installation, here are some suggestions for what you can explore next:

*   **Dive into the AI Demos:** If you haven't already, run the **[AI Model Gallery Web Application](./webapp_gallery_guide.md)** to see several NeuroFlux models in action.
*   **Understand the Big Picture:** Learn more about the project's goals and philosophy in the **[Introduction to NeuroFlux](./introduction.md)** and the **[Project Manifesto](./manifesto.md)**.
*   **Explore the Codebase:** Get an overview of how the project is organized by reading the **[Project Architecture document](./architecture.md)**.
*   **Discover Available Models:** Browse the **[AI Model Catalog](./model_catalog.md)** to learn about the different models in NeuroFlux, what they do, and how they can be used.
*   **Train AI Models:**
    *   For existing models with pre-defined training scripts, see the **[Guide: Training Existing Models](./training_existing_models_guide.md)**.
    *   To understand the framework for training advanced conceptual systems, refer to the **[Guide: Training Conceptual Systems](./training_conceptual_systems_guide.md)**.
*   **Delve into Core Concepts:** Explore the advanced ideas behind NeuroFlux:
    *   [Digital Pheromones](./concepts_digital_pheromones.md)
    *   [Autopoietic Systems & Self-Repair](./concepts_autopoietic_systems.md)
    *   [Advanced Security Mechanisms](./concepts_advanced_security.md)
    *   [Bio-Mimicry and Advanced Innovations](./concepts_bio_mimicry_advanced.md)
*   **Learn About Deployment:**
    *   [Deployment Overview](./deployment_overview.md)
    *   [Deploying Web Application with Docker](./deployment_webapp_docker.md)
    *   [Android Deployment Guide](./android_deployment.md)
*   **Contribute to the Project:** If you're interested in contributing, please read our **[Contributing Guide](../CONTRIBUTING.md)**.

We encourage you to explore these documents based on your interests. Welcome to the NeuroFlux community!
