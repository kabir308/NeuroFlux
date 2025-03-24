import torch
import torch.nn as nn
from typing import List, Dict, Any
from src.models.base_nano_model import BaseNanoModel

class MacroIA(nn.Module):
    """Orchestrateur principal de l'architecture Micro/Macro"""
    
    def __init__(self, micro_models: List[BaseNanoModel], fusion_strategy: str = "attention"):
        """
        Args:
            micro_models: Liste des nano-modèles à orchestrer
            fusion_strategy: Stratégie de fusion des résultats
        """
        super().__init__()
        self.micro_models = nn.ModuleList(micro_models)
        self.fusion_strategy = fusion_strategy
        
        # Initialisation du mécanisme de fusion
        self._init_fusion_layer()
        
    def _init_fusion_layer(self) -> None:
        """Initialise la couche de fusion selon la stratégie choisie"""
        if self.fusion_strategy == "attention":
            # Attention adaptative pour la fusion
            self.fusion = nn.Sequential(
                nn.Linear(sum(model.get_model_size() for model in self.micro_models), 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, len(self.micro_models))
            )
        else:
            # Fusion simple par concaténation
            self.fusion = nn.Linear(
                sum(model.get_model_size() for model in self.micro_models),
                len(self.micro_models)
            )
    
    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        """
        Args:
            inputs: Dictionnaire contenant les entrées pour chaque nano-modèle
            
        Returns:
            output: Résultat final après fusion
        """
        # Exécution parallèle des nano-modèles
        micro_outputs = []
        for model, input_data in zip(self.micro_models, inputs.values()):
            output = model(input_data)
            micro_outputs.append(output)
        
        # Fusion des résultats
        fused_output = self._fuse_outputs(micro_outputs)
        return fused_output
    
    def _fuse_outputs(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """Fusionne les résultats des nano-modèles"""
        if self.fusion_strategy == "attention":
            # Fusion avec attention
            stacked = torch.stack(outputs, dim=1)
            attention_weights = torch.softmax(self.fusion(stacked.mean(dim=1)), dim=1)
            return (stacked * attention_weights.unsqueeze(-1)).sum(dim=1)
        else:
            # Fusion simple par concaténation
            return torch.cat(outputs, dim=1)
    
    def optimize(self) -> None:
        """Optimise l'orchestrateur et ses nano-modèles"""
        for model in self.micro_models:
            model.optimize()
        
        # Optimisation de la couche de fusion
        self.fusion = torch.quantization.quantize_dynamic(
            self.fusion, {nn.Linear}, dtype=torch.qint8
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retourne l'état du système"""
        return {
            'total_size': sum(model.get_model_size() for model in self.micro_models),
            'num_models': len(self.micro_models),
            'fusion_strategy': self.fusion_strategy,
            'model_metadata': [model.get_metadata() for model in self.micro_models]
        }
