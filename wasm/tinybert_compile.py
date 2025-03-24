import torch
import torch.onnx
from models.tinybert.model import TinyBERT
import onnx
import onnxruntime as ort
from pathlib import Path

class TinyBERTCompiler:
    """Compilateur TinyBERT vers WebAssembly"""
    
    def __init__(self, model_path: str = None):
        """
        Args:
            model_path: Chemin vers le modèle pré-entraîné
        """
        self.model_path = model_path
        self.output_dir = Path("./wasm")
        self.output_dir.mkdir(exist_ok=True)
    
    def compile_to_onnx(self) -> str:
        """
        Compile le modèle PyTorch vers ONNX
        
        Returns:
            str: Chemin vers le fichier ONNX
        """
        # Charger le modèle
        model = TinyBERT()
        if self.model_path:
            model.load_state_dict(torch.load(self.model_path))
        
        # Préparer les données d'exemple
        dummy_input = torch.randint(0, 30000, (1, 128))
        attention_mask = torch.ones((1, 128))
        
        # Exporter vers ONNX
        onnx_path = self.output_dir / "tinybert.onnx"
        torch.onnx.export(
            model,
            (dummy_input, attention_mask),
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['last_hidden_state', 'pooler_output'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        return str(onnx_path)
    
    def optimize_onnx(self, onnx_path: str) -> str:
        """
        Optimise le modèle ONNX
        
        Args:
            onnx_path: Chemin vers le fichier ONNX
            
        Returns:
            str: Chemin vers le fichier ONNX optimisé
        """
        # Charger le modèle ONNX
        model = onnx.load(onnx_path)
        
        # Optimiser
        model = onnx.shape_inference.infer_shapes(model)
        model = onnx.optimizer.optimize(model)
        
        # Sauvegarder
        optimized_path = self.output_dir / "tinybert_optimized.onnx"
        onnx.save(model, optimized_path)
        
        return str(optimized_path)
    
    def compile_to_wasm(self, onnx_path: str) -> str:
        """
        Compile le modèle ONNX vers WebAssembly
        
        Args:
            onnx_path: Chemin vers le fichier ONNX
            
        Returns:
            str: Chemin vers le fichier WASM
        """
        # Compiler avec TVM
        import tvm
        from tvm import relay
        
        # Charger le modèle ONNX
        model, params = relay.frontend.from_onnx(
            onnx.load_model(onnx_path),
            {'input_ids': (1, 128), 'attention_mask': (1, 128)}
        )
        
        # Compiler
        target = "wasm32"
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(model, target, params=params)
        
        # Sauvegarder
        wasm_path = self.output_dir / "tinybert.wasm"
        lib.export_library(wasm_path)
        
        return str(wasm_path)
    
    def compile_all(self) -> str:
        """
        Compile le modèle complet
        
        Returns:
            str: Chemin vers le fichier WASM final
        """
        print("\n1. Compilation vers ONNX...")
        onnx_path = self.compile_to_onnx()
        
        print("\n2. Optimisation ONNX...")
        optimized_path = self.optimize_onnx(onnx_path)
        
        print("\n3. Compilation vers WebAssembly...")
        wasm_path = self.compile_to_wasm(optimized_path)
        
        print(f"\nCompilation terminée ! Fichier WASM : {wasm_path}")
        return wasm_path

def main():
    compiler = TinyBERTCompiler()
    compiler.compile_all()

if __name__ == "__main__":
    main()
