import torch
import numpy as np
from src.models.emotion_detector import EmotionDetector
from src.models.quantum_nano_model import QuantumNanoModel
from src.engines.macro_ia import MacroIA
from src.engines.hyperdimensional_engine import HyperdimensionalEngine
from src.engines.darwinian_optimizer import DarwinianOptimizer

def main():
    # Création des nano-modèles
    emotion_detector = EmotionDetector()
    quantum_model = QuantumNanoModel(num_qubits=4, num_layers=2)
    
    # Optimisation darwinienne
    optimizer = DarwinianOptimizer(EmotionDetector)
    optimized_model = optimizer.optimize(generations=10)
    
    # Création de l'orchestrateur Macro IA
    macro_ia = MacroIA([emotion_detector, quantum_model], fusion_strategy="attention")
    
    # Initialisation du moteur hyperdimensionnel
    hyper_engine = HyperdimensionalEngine()
    
    # Exemple d'utilisation
    text = ["Je suis vraiment heureux aujourd'hui!", "Cette situation est stressante."]
    image = np.random.rand(64, 64, 3)  # Exemple d'image
    
    # Encodage hyperdimensionnel
    text_hv = hyper_engine.encode(text[0], 'text')
    image_hv = hyper_engine.encode(image, 'image')
    
    # Fusion des représentations
    fused_hv = hyper_engine.fuse([text_hv, image_hv])
    
    # Simulation de la prédiction
    with torch.no_grad():
        predictions, confidence_scores = emotion_detector.predict(text, None)
        
        print("\nRésultats de la détection d'émotions:")
        for i, pred in enumerate(predictions):
            print(f"Texte: {text[i]}")
            print(f"Émotion prédite: {pred}")
            print(f"Score de confiance: {confidence_scores[i].max().item():.4f}")
            print("-" * 50)
    
    # Statut du système
    system_status = macro_ia.get_system_status()
    print("\nÉtat du système:")
    for key, value in system_status.items():
        print(f"{key}: {value}")
    
    # Statut du moteur hyperdimensionnel
    print("\nStatut du moteur hyperdimensionnel:")
    memory_usage = hyper_engine.get_memory_usage()
    for key, value in memory_usage.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
