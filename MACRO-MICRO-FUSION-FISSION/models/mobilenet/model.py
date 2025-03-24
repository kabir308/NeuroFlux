import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

class MobileNetV2(nn.Module):
    """Version optimisée de MobileNetV2 pour l'IA embarquée"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        
        # Modèle de base
        self.model = mobilenet_v2(pretrained=True)
        
        # Optimisations
        self._apply_quantization()
        self._apply_pruning()
        
        # Adaptation pour la classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, num_classes)
        )
    
    def _apply_quantization(self):
        """Applique la quantification du modèle"""
        # Quantification des poids
        for param in self.parameters():
            param.data = torch.quantize_per_tensor(
                param.data, 
                scale=1.0, 
                zero_point=0, 
                dtype=torch.qint8
            )
    
    def _apply_pruning(self):
        """Applique le pruning des poids"""
        # Pruning des connexions faibles
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                torch.nn.utils.prune.l1_unstructured(
                    module, 
                    name='weight', 
                    amount=0.5  # 50% de poids supprimés
                )
    
    def forward(self, x):
        """
        Args:
            x: Batch d'images (batch_size, channels, height, width)
            
        Returns:
            outputs: Scores de classification
        """
        return self.model(x)
    
    def get_model_size(self) -> float:
        """Retourne la taille du modèle en Mo"""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        return param_size / (1024 * 1024)
    
    def optimize(self) -> None:
        """Optimise le modèle pour la taille et la performance"""
        # Quantification
        self._apply_quantization()
        
        # Pruning
        self._apply_pruning()
        
        # Fusion des couches
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['features.0.0', 'features.0.1']]
        )
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle optimisé"""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str) -> 'MobileNetV2':
        """Charge un modèle pré-entraîné"""
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
