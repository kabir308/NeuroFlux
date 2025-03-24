from typing import List, Tuple
import torch
import torch.nn as nn
from .base_nano_model import BaseNanoModel

class EmotionDetector(BaseNanoModel):
    """Nano-modèle spécialisé pour la détection d'émotions dans le texte"""
    
    def __init__(self, vocab_size: int = 1000, embedding_dim: int = 64, num_classes: int = 5):
        """
        Args:
            vocab_size: Taille du vocabulaire
            embedding_dim: Dimension des embeddings
            num_classes: Nombre de classes d'émotions
        """
        super().__init__(model_size=2)  # Cible 2 Mo
        self.task_type = "emotion_detection"
        
        # Architecture optimisée pour la taille
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, embedding_dim // 2, batch_first=True)
        self.fc = nn.Linear(embedding_dim // 2, num_classes)
        
    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text: Batch de textes (batch_size, seq_len)
        
        Returns:
            logits: Scores non-normalisés pour chaque classe d'émotion
        """
        embeddings = self.embed(text)
        _, (hidden, _) = self.lstm(embeddings)
        return self.fc(hidden[-1])
    
    def optimize(self) -> None:
        """Optimise spécifiquement pour la détection d'émotions"""
        # Quantification
        self.embed = torch.quantization.quantize_dynamic(
            self.embed, {torch.nn.Embedding}, dtype=torch.qint8
        )
        
        # Fusion des couches
        self.lstm.flatten_parameters()
    
    def predict(self, text: List[str], tokenizer) -> Tuple[List[str], torch.Tensor]:
        """
        Prédit les émotions pour une liste de textes
        
        Args:
            text: Liste de textes à analyser
            tokenizer: Tokenizer pour prétraiter le texte
            
        Returns:
            Tuple contenant:
            - Liste des émotions prédites
            - Scores de confiance
        """
        self.eval()
        with torch.no_grad():
            inputs = tokenizer(text, return_tensors="pt", padding=True)
            outputs = self.forward(inputs.input_ids)
            predictions = torch.argmax(outputs, dim=1)
            
        return predictions, torch.softmax(outputs, dim=1)
