import ast
import inspect
import numpy as np
from typing import List, Tuple, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class NanoAutoCoder:
    """Nano-IA capable d'auto-réparation via génération de code"""
    
    def __init__(self, model_size: str = "gpt2"):
        """
        Args:
            model_size: Taille du modèle GPT-2 à utiliser
        """
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_size)
        self.model = GPT2LMHeadModel.from_pretrained(model_size)
        self.sandbox = {
            "np": np,
            "allowed_functions": {
                "sqrt": np.sqrt,
                "mean": np.mean,
                "sum": np.sum,
                "max": np.max,
                "min": np.min
            }
        }
    
    def generate_code(self, task: str) -> str:
        """
        Génère du code Python pour une tâche spécifique
        
        Args:
            task: Description de la tâche à accomplir
            
        Returns:
            code: Code Python généré
        """
        prompt = f"""# Python function to {task}
import numpy as np
def solution(inputs):"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=200,
            temperature=0.7,
            num_return_sequences=1
        )
        return self.tokenizer.decode(outputs[0])
    
    def safe_execute(self, code: str, inputs: Any) -> Any:
        """
        Exécute du code dans un environnement sécurisé
        
        Args:
            code: Code Python à exécuter
            inputs: Données d'entrée pour la fonction
            
        Returns:
            result: Résultat de l'exécution
        """
        try:
            # Extraction de la fonction générée
            tree = ast.parse(code)
            func_def = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
            func_code = compile(tree, "<string>", "exec")
            
            # Exécution sécurisée
            exec_env = self.sandbox.copy()
            exec(func_code, exec_env)
            result = exec_env[func_def.name](inputs)
            return result
        except Exception as e:
            print(f"Erreur d'exécution: {str(e)}")
            return None
    
    def self_repair(self, task: str, test_cases: List[Tuple[Any, Any]], max_attempts: int = 3) -> str:
        """
        Tente de réparer une fonction via génération de code
        
        Args:
            task: Description de la tâche à accomplir
            test_cases: Liste de tuples (input, expected_output)
            max_attempts: Nombre maximum de tentatives
            
        Returns:
            code: Code fonctionnel généré
        """
        for attempt in range(max_attempts):
            print(f"Tentative {attempt + 1}/{max_attempts}")
            code = self.generate_code(task)
            
            try:
                results = [self.safe_execute(code, tc[0]) for tc in test_cases]
                if all(np.isclose(res, expected) for res, (_, expected) in zip(results, test_cases)):
                    print("Réparation réussie!")
                    return code
            except:
                continue
        
        return self.evolve_and_retry(task, test_cases)
    
    def evolve_and_retry(self, task: str, test_cases: List[Tuple[Any, Any]]) -> str:
        """
        Évolution du code via mutation et croisement
        
        Args:
            task: Description de la tâche
            test_cases: Cas de test
            
        Returns:
            code: Code évolué
        """
        # Génération de plusieurs versions
        codes = [self.generate_code(task) for _ in range(5)]
        
        # Évaluation des performances
        scores = []
        for code in codes:
            try:
                results = [self.safe_execute(code, tc[0]) for tc in test_cases]
                score = sum(1 for res, (_, expected) in zip(results, test_cases) 
                          if np.isclose(res, expected))
                scores.append(score)
            except:
                scores.append(0)
        
        # Sélection des meilleurs codes
        best_codes = [code for code, score in zip(codes, scores) if score > 0]
        
        if not best_codes:
            return self.generate_code(task)  # Nouvelle tentative
        
        # Croisement des meilleurs codes
        best_code = self._cross_codes(best_codes)
        return best_code
    
    def _cross_codes(self, codes: List[str]) -> str:
        """Croise plusieurs codes pour créer une nouvelle version"""
        if not codes:
            return ""
            
        # Sélection aléatoire de deux codes
        code1 = random.choice(codes)
        code2 = random.choice(codes)
        
        # Croisement simple
        split_point = random.randint(0, len(code1))
        new_code = code1[:split_point] + code2[split_point:]
        
        # Mutation
        if random.random() < 0.1:  # 10% de chance de mutation
            new_code = self._mutate_code(new_code)
        
        return new_code
    
    def _mutate_code(self, code: str) -> str:
        """Mutate le code en ajoutant/modifiant des éléments"""
        # Implémentation simple de mutation
        mutations = [
            lambda x: x.replace("np.sqrt", "np.power(x, 0.5)"),
            lambda x: x.replace("np.sum", "np.mean"),
            lambda x: x + "\n    result *= 2"  # Exemple de mutation simple
        ]
        
        for mutation in mutations:
            if random.random() < 0.3:  # 30% de chance pour chaque mutation
                code = mutation(code)
        
        return code
