import unittest
import torch
from models.tinybert.model import TinyBERT
from models.mobilenet.model import MobileNetV2

class TestTinyBERT(unittest.TestCase):
    """Tests pour le modèle TinyBERT"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.model = TinyBERT()
        self.input_ids = torch.randint(0, 30000, (2, 128))
        self.attention_mask = torch.ones((2, 128))
    
    def test_forward_pass(self):
        """Test du passage avant"""
        outputs = self.model(self.input_ids, self.attention_mask)
        self.assertIn('last_hidden_state', outputs)
        self.assertIn('pooler_output', outputs)
    
    def test_model_size(self):
        """Test de la taille du modèle"""
        size = self.model.get_model_size()
        self.assertLess(size, 2.0)  # Moins de 2 Mo
    
    def test_optimization(self):
        """Test de l'optimisation"""
        initial_size = self.model.get_model_size()
        self.model.optimize()
        optimized_size = self.model.get_model_size()
        self.assertLess(optimized_size, initial_size)

class TestMobileNetV2(unittest.TestCase):
    """Tests pour le modèle MobileNetV2"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.model = MobileNetV2()
        self.input = torch.randn((2, 3, 224, 224))
    
    def test_forward_pass(self):
        """Test du passage avant"""
        outputs = self.model(self.input)
        self.assertEqual(outputs.shape[1], 1000)  # 1000 classes
    
    def test_model_size(self):
        """Test de la taille du modèle"""
        size = self.model.get_model_size()
        self.assertLess(size, 2.0)  # Moins de 2 Mo
    
    def test_optimization(self):
        """Test de l'optimisation"""
        initial_size = self.model.get_model_size()
        self.model.optimize()
        optimized_size = self.model.get_model_size()
        self.assertLess(optimized_size, initial_size)

class TestModelIntegration(unittest.TestCase):
    """Tests d'intégration pour les modèles"""
    
    def setUp(self):
        """Initialisation avant chaque test"""
        self.tinybert = TinyBERT()
        self.mobilenet = MobileNetV2()
    
    def test_multi_model_pipeline(self):
        """Test d'un pipeline multi-modèles"""
        # Test TinyBERT
        input_ids = torch.randint(0, 30000, (2, 128))
        attention_mask = torch.ones((2, 128))
        bert_outputs = self.tinybert(input_ids, attention_mask)
        
        # Test MobileNet
        images = torch.randn((2, 3, 224, 224))
        mobilenet_outputs = self.mobilenet(images)
        
        self.assertIsNotNone(bert_outputs)
        self.assertIsNotNone(mobilenet_outputs)

if __name__ == '__main__':
    unittest.main()
