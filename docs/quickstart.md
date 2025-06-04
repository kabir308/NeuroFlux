# 🚀 Démarrage Rapide NeuroFlux

## 🌍 En Français

Bienvenue dans le guide de démarrage rapide de NeuroFlux ! Suivez ces étapes simples pour commencer à utiliser notre framework d'IA microscopique.

### 📋 Prérequis

- Python 3.10 ou supérieur
- Git
- Un environnement virtuel Python

### 📦 Installation

1. **Clonez le Repository**
   ```bash
   git clone https://github.com/neuroflux/neuroflux.git
   cd neuroflux
   ```

2. **Créez un Environnement Virtuel**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Installez les Dépendances**
   ```bash
   pip install -r requirements.txt
   ```

### 🧠 Accéder aux Modèles NeuroFlux Pré-Optimisés

NeuroFlux fournit une suite de modèles IA pré-optimisés pour le déploiement sur des appareils périphériques (edge).
Vous pouvez préparer ces modèles en utilisant l'utilitaire `NeuroFluxDataset` du module `huggingface`.

Assurez-vous d'abord d'avoir les bibliothèques nécessaires pour la conversion et l'optimisation des modèles :
```bash
pip install -r huggingface/requirements.txt
# Vous pourriez également avoir besoin d'installer des outils de conversion spécifiques comme onnx-tensorflow ou paddle2onnx
# pip install onnx onnx-tf paddle2onnx # Exemple
```

Ensuite, vous pouvez utiliser le script Python suivant pour générer les modèles optimisés :

```python
from huggingface.neuroflux import NeuroFluxDataset
import os

# Créez une instance pour tous les modèles (ou un spécifique)
# Le 'name' dans NeuroFluxConfig déterminera quels modèles sont traités.
# Pour cet exemple, supposons que vous ayez une configuration dans neuroflux.py
# qui traite un modèle comme 'mobilenet' ou 'efficientnet-lite'.

# Exemple : Traiter le type de modèle 'mobilenet'
print("Préparation du modèle NeuroFlux MobileNet...")
# Assurez-vous que "mobilenet" est un nom de configuration valide dans NeuroFluxDataset.BUILDER_CONFIGS
dataset_builder = NeuroFluxDataset(name="mobilenet")

# Définir un répertoire de sortie pour le dataset
output_dir = "./neuroflux_optimized_models_fr" # Répertoire de sortie spécifique à la langue
os.makedirs(output_dir, exist_ok=True)

# Ceci va télécharger, convertir et optimiser les modèles spécifiés par la configuration
# Les modèles TFLite résultants et les métadonnées seront dans 'output_dir'
dataset_builder.download_and_prepare(download_dir=output_dir)

print(f"Les modèles optimisés et les métadonnées sont disponibles dans : {output_dir}")
print("Vous pouvez trouver les fichiers de modèle .tflite dans les sous-répertoires respectifs des modèles.")
```

Cela inclut une variété de modèles (par exemple, pour la vision comme EfficientNet-Lite et NanoDet-Plus, des modèles NLP comme FastBERT, et plus encore). Ces modèles sont automatiquement convertis en TFLite et peuvent être optimisés pour l'exécution GPU et la quantification hybride afin d'améliorer les performances sur les appareils mobiles/périphériques.

### 🚀 Exécuter l'Inférence avec un Modèle TFLite

Une fois que vous avez généré un modèle TFLite (par exemple, `mobilenet.tflite` dans le répertoire `output_dir/mobilenet/`), vous pouvez exécuter une inférence en utilisant TensorFlow Lite Interpreter. Assurez-vous d'avoir `tensorflow` et `numpy` installés (`pip install tensorflow numpy`).

```python
import tensorflow as tf
import numpy as np
import os

# Chemin vers votre modèle TFLite généré
# Exemple : './neuroflux_optimized_models_fr/mobilenet/mobilenet_optimized.tflite'
# Assurez-vous que ce chemin correspond à un modèle que vous avez généré
model_path = "./neuroflux_optimized_models_fr/mobilenet/mobilenet_optimized.tflite" # Adaptez ce chemin

if not os.path.exists(model_path):
    print(f"Modèle non trouvé à {model_path}. Veuillez d'abord générer les modèles.")
else:
    # Charger le modèle TFLite et allouer les tenseurs.
    interpreteur = tf.lite.Interpreter(model_path=model_path)
    interpreteur.allocate_tensors()

    # Obtenir les tenseurs d'entrée et de sortie.
    details_entree = interpreteur.get_input_details()
    details_sortie = interpreteur.get_output_details()

    # Préparer des données d'entrée factices
    # Cet exemple suppose une entrée image 1x224x224x3 pour un modèle comme MobileNet
    # Ajustez la forme (shape) et le type de données (dtype) selon les exigences de votre modèle
    forme_entree = details_entree[0]['shape']
    donnees_entree = np.array(np.random.random_sample(forme_entree), dtype=details_entree[0]['dtype'])
    interpreteur.set_tensor(details_entree[0]['index'], donnees_entree)

    # Exécuter l'inférence
    interpreteur.invoke()

    # Obtenir les données de sortie
    donnees_sortie = interpreteur.get_tensor(details_sortie[0]['index'])
    print("Données de sortie:", donnees_sortie)
    print("Inférence TFLite réussie !")
```

### 💡 Autres Solutions d'Inférence

Bien que `NeuroFluxDataset` génère principalement des modèles TFLite optimisés, d'autres options existent :
- Pour les modèles initialement au format ONNX, ou ceux convertibles en ONNX, vous pouvez utiliser **ONNX Runtime Mobile**.
- Pour des déploiements encore plus spécialisés, **MNN** et **NCNN** sont des moteurs d'inférence puissants ; vous pouvez explorer leurs outils pour convertir des modèles à leurs formats respectifs.

### 🏗️ Structure du Projet

```
neuroflux/
├── src/                  # Code source
│   ├── models/           # Nano-modèles
│   ├── engines/          # Moteurs d'orchestration
│   └── pheromones/      # Système de phéromones
├── wasm/                 # Compilation WebAssembly
├── docs/                # Documentation
└── tests/               # Tests
```

### 🚀 Premier Modèle

1. **Créez un Modèle TinyBERT**
   ```python
   from models.tinybert.model import TinyBERT
   
   # Créer un modèle
   model = TinyBERT()
   
   # Optimiser
   model.optimize()
   ```

2. **Compilez en WebAssembly**
   ```bash
   python wasm/tinybert_compile.py
   ```

### 🤖 Essaim de Phéromones

1. **Initialisez la Base de Données**
   ```python
   from src.pheromones.digital_pheromones import PheromoneDatabase
   
   # Initialiser la base
   db = PheromoneDatabase()
   ```

2. **Créez une Phéromone**
   ```python
   from src.pheromones.digital_pheromones import DigitalPheromone
   
   # Créer une phéromone
   pheromone = DigitalPheromone(
       agent_id="agent_1",
       signal_type="knowledge",
       data={"topic": "AI", "confidence": 0.9},
       ttl=3600  # 1 heure
   )
   
   # Ajouter à la base
   db.add_pheromone(pheromone)
   ```

### 🧪 Tests

Pour exécuter les tests :

```bash
pytest tests/ -v
```

### 📚 Documentation

La documentation complète est disponible sur GitHub Pages :

[https://neuroflux.github.io/neuroflux/](https://neuroflux.github.io/neuroflux/)

### 🤝 Contribuer

Voir [CONTRIBUTING.md](https://github.com/neuroflux/neuroflux/blob/main/CONTRIBUTING.md) pour savoir comment contribuer.

### 📝 Licence

Ce projet est sous licence Apache 2.0. Voir [LICENSE](https://github.com/neuroflux/neuroflux/blob/main/LICENSE) pour plus de détails.

## 🌐 In English

## 🚀 Quick Start NeuroFlux

### 📋 Prerequisites

- Python 3.10 or higher
- Git
- Python virtual environment

### 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/neuroflux/neuroflux.git
   cd neuroflux
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### 🧠 Accessing Pre-Optimized NeuroFlux Models

NeuroFlux provides a suite of AI models pre-optimized for edge deployment.
You can prepare these models using the `NeuroFluxDataset` utility from the `huggingface` module.

First, ensure you have the necessary libraries for model conversion and optimization:
```bash
pip install -r huggingface/requirements.txt
# You might also need to install specific conversion tools like onnx-tensorflow or paddle2onnx
# pip install onnx onnx-tf paddle2onnx # Example
```

Then, you can use the following Python script to generate the optimized models:

```python
from huggingface.neuroflux import NeuroFluxDataset
import os

# Create an instance for all models (or a specific one)
# The 'name' in NeuroFluxConfig will determine which models are processed.
# For this example, let's assume you have a config in neuroflux.py
# that processes a model like 'mobilenet' or 'efficientnet-lite'.

# Example: Process the 'mobilenet' model type
print("Preparing NeuroFlux MobileNet model...")
# Ensure "mobilenet" is a valid config name from NeuroFluxDataset.BUILDER_CONFIGS
dataset_builder = NeuroFluxDataset(name="mobilenet")

# Define an output directory for the dataset
output_dir = "./neuroflux_optimized_models_en" # Language-specific output directory
os.makedirs(output_dir, exist_ok=True)

# This will download, convert, and optimize the models specified by the config
# The resulting TFLite models and metadata will be in 'output_dir'
dataset_builder.download_and_prepare(download_dir=output_dir)

print(f"Optimized models and metadata are available in: {output_dir}")
print("You can find .tflite model files in the respective model subdirectories.")
```

This includes a variety of models (e.g., for vision like EfficientNet-Lite and NanoDet-Plus, NLP models like FastBERT, and more). These models are automatically converted to TFLite and can be optimized for GPU execution and hybrid quantization for enhanced performance on mobile/edge devices.

### 🚀 Running Inference with a TFLite Model

Once you have a TFLite model generated (e.g., `mobilenet_optimized.tflite` in the `output_dir/mobilenet/` directory), you can run inference using the TensorFlow Lite Interpreter. Ensure you have `tensorflow` and `numpy` installed (`pip install tensorflow numpy`).

```python
import tensorflow as tf
import numpy as np
import os

# Path to your generated TFLite model
# Example: './neuroflux_optimized_models_en/mobilenet/mobilenet_optimized.tflite'
# Ensure this path matches a model you have generated
model_path = "./neuroflux_optimized_models_en/mobilenet/mobilenet_optimized.tflite" # Adjust this path

if not os.path.exists(model_path):
    print(f"Model not found at {model_path}. Please generate models first.")
else:
    # Load the TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare dummy input data
    # This example assumes a 1x224x224x3 image input for a model like MobileNet
    # Adjust the shape and dtype according to your model's input requirements
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("Output data:", output_data)
    print("TFLite Inference successful!")
```

### 💡 Other Inference Solutions

The `NeuroFluxDataset` primarily generates optimized TFLite models. For models originally in ONNX format, or those convertible to ONNX, you can also use **ONNX Runtime Mobile**. For even more specialized deployments, **MNN** and **NCNN** are powerful inference engines; you can explore their tools to convert models to their respective formats.

### 🏗️ Project Structure

```
neuroflux/
├── src/                  # Source code
│   ├── models/           # Nano-models
│   ├── engines/          # Orchestration engines
│   └── pheromones/      # Pheromone system
├── wasm/                 # WebAssembly compilation
├── docs/                # Documentation
└── tests/               # Tests
```

### 🚀 First Model

1. **Create a TinyBERT Model**
   ```python
   from models.tinybert.model import TinyBERT
   
   # Create a model
   model = TinyBERT()
   
   # Optimize
   model.optimize()
   ```

2. **Compile to WebAssembly**
   ```bash
   python wasm/tinybert_compile.py
   ```

### 🤖 Pheromone Swarm

1. **Initialize the Database**
   ```python
   from src.pheromones.digital_pheromones import PheromoneDatabase
   
   # Initialize the database
   db = PheromoneDatabase()
   ```

2. **Create a Pheromone**
   ```python
   from src.pheromones.digital_pheromones import DigitalPheromone
   
   # Create a pheromone
   pheromone = DigitalPheromone(
       agent_id="agent_1",
       signal_type="knowledge",
       data={"topic": "AI", "confidence": 0.9},
       ttl=3600  # 1 hour
   )
   
   # Add to database
   db.add_pheromone(pheromone)
   ```

### 🧪 Tests

To run the tests:

```bash
pytest tests/ -v
```

### 📚 Documentation

The complete documentation is available on GitHub Pages:

[https://neuroflux.github.io/neuroflux/](https://neuroflux.github.io/neuroflux/)

### 🤝 Contributing

See [CONTRIBUTING.md](https://github.com/neuroflux/neuroflux/blob/main/CONTRIBUTING.md) to learn how to contribute.

### 📝 License

This project is under Apache 2.0 license. See [LICENSE](https://github.com/neuroflux/neuroflux/blob/main/LICENSE) for details.
