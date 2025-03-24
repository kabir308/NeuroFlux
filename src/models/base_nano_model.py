import torch.nn as nn
from typing import Any, Dict, List

class BaseNanoModel(nn.Module):
    """Classe de base pour les nano-modèles (Micro IA)"""
    
    def __init__(self, model_size: int = 5):
        """
        Args:
            model_size: Taille cible du modèle en Mo
        """
        super().__init__()
        self.model_size = model_size
        self.task_type = None  # À définir par les classes filles
        
    def forward(self, x: Any) -> Any:
        """Méthode à implémenter par les classes filles"""
        raise NotImplementedError
    
    def get_model_size(self) -> float:
        """Retourne la taille du modèle en Mo"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)
    
    def optimize(self) -> None:
        """Optimise le modèle pour la taille et la performance"""
        # À implémenter selon les besoins spécifiques
        pass
    
    def get_metadata(self) -> Dict[str, Any]:
        """Retourne les métadonnées du modèle"""
        return {
            'model_size': self.get_model_size(),
            'task_type': self.task_type,
            'architecture': self.__class__.__name__
        }
