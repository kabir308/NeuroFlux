# NeuroFlux Project Architecture

This document provides an overview of the NeuroFlux project's architecture, detailing its main components, their purposes, and how they interact. Understanding this architecture is key for developers, contributors, and anyone looking to delve deeper into the project's technical aspects.

## High-Level Overview

NeuroFlux is designed as a modular and extensible framework to support the development and deployment of microscopic, autonomous, and intelligent AI systems. Conceptually, it can be viewed in layers:

1.  **Core Intelligence & Logic Layer (`src/`):** This is the brain of NeuroFlux, containing the Python implementations for its advanced AI capabilities, including digital pheromones, autopoietic self-healing, advanced security mechanisms, and other bio-inspired systems.
2.  **Model Development & Optimization Layer (`huggingface/`, `models/`):** This layer focuses on creating, training, and optimizing AI models to be hyper-efficient for edge deployment. It includes scripts for model training, conversion (e.g., to TFLite), and a catalog of models.
3.  **Application & Demonstration Layer (`webapp/`):** This layer provides tangible examples and interactive demos of NeuroFlux concepts and models, primarily through the AI Model Gallery web application.
4.  **Embedded & Hardware Integration Layer (`embedded/`, `wasm/`):** This layer deals with deploying and running NeuroFlux components on specific hardware, such as ESP32 microcontrollers, and compiling models to formats like WebAssembly for on-device inference.
5.  **Tooling & Utilities Layer (`scripts/`, `tests/`):** This layer contains utility scripts for tasks like master training orchestration, testing frameworks, and other developer tools.
6.  **Documentation Layer (`docs/`):** Contains all project documentation (you are here!).

<!-- TODO: Insert a high-level architecture diagram here (e.g., a box and arrow diagram showing these layers and their primary interactions). -->

## Core Components Details

### 1. `src/` - Core NeuroFlux Engine
This directory houses the primary Python source code that defines the unique capabilities of NeuroFlux.

*   **`src/autopoietic/`**: Implements self-healing, self-optimization, and potentially self-generation capabilities.
    *   `self_healing_ai.py`: Manages model checkpointing, degradation detection, and regeneration.
    *   `predictive_failure_analyzer.py`: Aims to predict component failures before they occur.
    *   `nano_autocoder.py`: (More experimental) Explores AI-driven code generation/repair.
    *   `force_field_agent.py`: (Purpose less defined from current context, may be for environment interaction or agent integrity).
*   **`src/pheromones/`**: Contains the logic for digital pheromone communication.
    *   `digital_pheromones.py`: Defines `DigitalPheromone`, `PheromoneDatabase`, and `PheromoneNetwork` classes for creating, storing, and broadcasting virtual pheromones for swarm coordination.
*   **`src/security/`**: Placeholder for implementing advanced security mechanisms.
    *   `decoy_pheromones.py`: System for generating false traffic and honeypot signals.
    *   `defensive_mutation_engine.py`: Engine for triggering code mutations in response to threats.
    *   `README.md`: Outlines planned security concepts (quantum encryption, bio-auth, etc.).
*   **`src/advanced_systems/`**: For implementing cutting-edge, bio-inspired, and potentially disruptive innovations.
    *   `digital_mitosis.py`: Agent capable of self-replication and fragmentation.
    *   `quantum_synapses.py`: Conceptual simulation of quantum-entangled communication and holographic memory.
    *   `README.md`: Outlines the scope for these advanced systems.
*   **`src/models/`**: (Root level `src/models/`) Contains some base model definitions or interfaces.
    *   `base_nano_model.py`: Likely a base class for NeuroFlux's microscopic models.
    *   `emotion_detector.py`: (Potentially a model definition, distinct from the demo in webapp).
    *   `quantum_nano_model.py`: Conceptual model integrating quantum principles.
*   **`src/engines/`**: Intended for orchestration and higher-level control logic.
    *   `darwinian_optimizer.py`: Suggests evolutionary algorithms for optimization.
    *   `hyperdimensional_engine.py`: Hints at using Hyperdimensional Computing concepts.
    *   `macro_ia.py`: (Purpose less defined, perhaps for large-scale coordination).

### 2. `huggingface/` - Model Management & Optimization
This directory is central to creating, training, and preparing models for NeuroFlux, leveraging Hugging Face tools and concepts.

*   **`huggingface/models/`**: Contains specific AI model implementations, their training scripts (if applicable), and READMEs.
    *   Examples: `emotion-detector/`, `mobilenet/`, `tinybert/` (with `train.py`), `efficientnet-lite/`, `fastbert/` (with template `train.py`), `nanodet-plus/`, etc.
*   **`huggingface/neuroflux.py`**: (Mentioned in `docs/quickstart.md`) Likely contains the `NeuroFluxDataset` class for generating TFLite and other optimized model formats.
*   **`huggingface/app.py`**: Potentially a Gradio or Streamlit app for Hugging Face Space demos.
*   `huggingface/requirements.txt`: Python dependencies specific to these tasks.

### 3. `webapp/` - AI Model Gallery & Demos
This directory contains the Flask-based web application that showcases various NeuroFlux models and concepts.

*   `app.py`: The main Flask application file.
*   `templates/`: HTML templates for the web pages.
*   `static/`: CSS, JavaScript, and other static assets.
*   `llm_client.py`, `llm_config.py`: For the LLM demo.
*   `ncnn_object_detector.py`: For the NanoDet-Plus NCNN object detection demo.
*   `webapp/models/`: Contains pre-downloaded or placeholder model files (TFLite, NCNN params/bin) specifically for the web demos.

### 4. `scripts/` - Utility and Training Scripts
This directory holds higher-level scripts for managing the project.

*   `train_existing_models.py`: Master script to discover and run training for models in `huggingface/models/` that have `train.py` files.
*   `train_conceptual_systems.py`: Framework script with placeholders to guide the development of training pipelines for advanced conceptual NeuroFlux systems.

### 5. `embedded/esp32/` - ESP32 Microcontroller Integration
Contains C++ code and resources for ESP32 development.

*   `swarm_communication.cpp`: Initial C++ skeleton for setting up multi-modal (WiFi, LoRa, Bluetooth Mesh) communication for pheromone exchange between ESP32 devices.
*   `README.md`: Outlines purpose and planned development for ESP32 components.

### 6. `wasm/` - WebAssembly Compilation
Focuses on compiling models to WebAssembly for client-side/on-device browser inference.

*   `tinybert_compile.py`: Example script for compiling TinyBERT to Wasm (faced environment limitations as per root `README.md`).
*   `Makefile`, `cross_compile.py`: Other utilities related to Wasm compilation.

### 7. `tests/` - Automated Tests
Contains automated tests for the project, likely using PyTest.

*   `unit/`: For unit tests of individual modules/functions.
*   `webapp/`: For tests specific to the Flask web application.
*   *(Refer to `docs/testing_guide.md` for details on running and writing tests).*

### 8. `docs/` - Documentation
Houses all project documentation, including developer guides, concept explanations, and user manuals.

*   `index.md`: The main table of contents for all documentation.
*   `business/`: Subdirectory for business-related documents like the pitch deck outline.

## Interactions and Data Flow (Conceptual)

*   **Model Development:** Models are defined/adapted in `huggingface/models/` or `src/models/`. Training occurs using scripts (either directly or via `scripts/train_existing_models.py`).
*   **Optimization & Deployment:** `huggingface/neuroflux.py` (or similar tools) converts trained models to efficient formats (e.g., TFLite). These can be used by the `webapp/`, deployed to Android (see `docs/android_deployment.md`), or targeted for `embedded/` or `wasm/`.
*   **Core Logic Execution:** The `src/` modules provide the runtime intelligence. For instance, an agent running on an ESP32 (`embedded/`) might use `src/pheromones/` logic (ported or called via an API) to communicate. The `webapp/` might demonstrate a Python-based simulation of these core concepts.
*   **Self-Healing Loop:** `src/autopoietic/predictive_failure_analyzer.py` might monitor system health (data could come from anywhere), `src/autopoietic/self_healing_ai.py` would manage checkpoints and recovery, potentially triggering `src/autopoietic/nano_autocoder.py` or `src/security/defensive_mutation_engine.py` for more advanced repair/defense.
*   **Pheromone Network:** Agents (whether simulated in Python, running in the `webapp`, or on actual `embedded/` devices) would use the `PheromoneNetwork` to broadcast and receive signals, influencing collective behavior.

This architecture is designed to be modular, allowing different components to be developed, tested, and deployed somewhat independently while contributing to the overall NeuroFlux vision.
