import os
import subprocess
from typing import List, Dict

class WASMCompiler:
    """Compilateur croisé pour WebAssembly"""
    
    def __init__(self, models_dir: str = "../models"):
        """
        Args:
            models_dir: Dossier contenant les modèles à compiler
        """
        self.models_dir = models_dir
        self.output_dir = "./wasm"
        
        # Création du dossier de sortie
        os.makedirs(self.output_dir, exist_ok=True)
    
    def compile_model(self, model_path: str) -> str:
        """
        Compile un modèle en WebAssembly
        
        Args:
            model_path: Chemin vers le modèle à compiler
            
        Returns:
            output_path: Chemin vers le fichier WASM compilé
        """
        # Extraction du nom du modèle
        model_name = os.path.basename(model_path)
        output_path = os.path.join(self.output_dir, f"{os.path.splitext(model_name)[0]}.wasm")
        
        try:
            # Compilation avec Emscripten
            subprocess.run([
                "emcc",
                model_path,
                "-o",
                output_path,
                "-s",
                "WASM=1",
                "-s",
                "MODULARIZE=1",
                "-s",
                "EXPORT_NAME=\"Module\""
            ], check=True)
            
            return output_path
            
        except subprocess.CalledProcessError as e:
            print(f"Erreur de compilation: {e}")
            return None
    
    def compile_all_models(self) -> List[str]:
        """Compile tous les modèles dans le dossier"""
        compiled_models = []
        
        # Liste des fichiers de modèles
        model_files = [
            f for f in os.listdir(self.models_dir)
            if f.endswith(('.onnx', '.pt'))
        ]
        
        for model_file in model_files:
            model_path = os.path.join(self.models_dir, model_file)
            compiled_path = self.compile_model(model_path)
            if compiled_path:
                compiled_models.append(compiled_path)
        
        return compiled_models
    
    def optimize_size(self, wasm_file: str) -> None:
        """
        Optimise la taille du fichier WASM
        
        Args:
            wasm_file: Chemin vers le fichier WASM à optimiser
        """
        try:
            # Utilisation de wasm-opt pour l'optimisation
            subprocess.run([
                "wasm-opt",
                "-Oz",
                wasm_file,
                "-o",
                wasm_file
            ], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"Erreur d'optimisation: {e}")
    
    def get_size_report(self) -> Dict[str, float]:
        """Génère un rapport de taille pour les fichiers WASM"""
        size_report = {}
        
        for wasm_file in os.listdir(self.output_dir):
            if wasm_file.endswith('.wasm'):
                file_path = os.path.join(self.output_dir, wasm_file)
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                size_report[wasm_file] = size_mb
        
        return size_report

def main():
    compiler = WASMCompiler()
    compiled_models = compiler.compile_all_models()
    
    print("\nModèles compilés:")
    for model in compiled_models:
        print(f"- {model}")
    
    # Optimisation des fichiers WASM
    for wasm_file in compiled_models:
        compiler.optimize_size(wasm_file)
    
    # Rapport de taille
    size_report = compiler.get_size_report()
    print("\nRapport de taille:")
    for name, size in size_report.items():
        print(f"{name}: {size:.2f} Mo")

if __name__ == "__main__":
    main()
