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
