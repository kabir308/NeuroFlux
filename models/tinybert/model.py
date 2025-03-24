import torch
import torch.nn as nn
from transformers import BertConfig, BertModel

class TinyBERT(nn.Module):
    """Version optimisée de TinyBERT pour l'IA embarquée"""
    
    def __init__(self, config=None):
        super().__init__()
        
        # Configuration optimisée
        if config is None:
            config = BertConfig(
                vocab_size=30000,
                hidden_size=128,
                num_hidden_layers=4,
                num_attention_heads=2,
                intermediate_size=512,
                max_position_embeddings=512,
                type_vocab_size=2,
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
            )
        
        # Modèle de base
        self.bert = BertModel(config)
        
        # Optimisations
        self._apply_quantization()
        self._apply_pruning()
    
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
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(
                    module, 
                    name='weight', 
                    amount=0.5  # 50% de poids supprimés
                )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Batch de tokens (batch_size, seq_len)
            attention_mask: Masque d'attention (batch_size, seq_len)
            
        Returns:
            outputs: Dictionnaire contenant les sorties
        """
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Récupération des embeddings
        return {
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output
        }
    
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
        self.bert = torch.quantization.fuse_modules(
            self.bert,
            [['layer.0.attention.self.query', 'layer.0.attention.self.key']]
        )
    
    def save(self, path: str) -> None:
        """Sauvegarde le modèle optimisé"""
        torch.save(self.state_dict(), path)
    
    @classmethod
    def load(cls, path: str) -> 'TinyBERT':
        """Charge un modèle pré-entraîné"""
        model = cls()
        model.load_state_dict(torch.load(path))
        return model
