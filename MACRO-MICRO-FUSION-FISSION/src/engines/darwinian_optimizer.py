import numpy as np
from deap import base, creator, tools, algorithms
from typing import List, Dict, Any, Type
from src.models.base_nano_model import BaseNanoModel

class DarwinianOptimizer:
    """Optimiseur darwinien pour l'évolution des nano-modèles"""
    
    def __init__(self, model_class: Type[BaseNanoModel], population_size: int = 50):
        """
        Args:
            model_class: Classe du nano-modèle à optimiser
            population_size: Taille de la population
        """
        self.model_class = model_class
        self.population_size = population_size
        
        # Initialisation de DEAP
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        self.toolbox = base.Toolbox()
        self._init_toolbox()
    
    def _init_toolbox(self) -> None:
        """Initialise les outils de DEAP"""
        # Génération aléatoire de poids
        self.toolbox.register("weight", np.random.uniform, -1, 1)
        
        # Création d'un individu
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                           self.toolbox.weight, n=self.model_class().get_model_size())
        
        # Création de la population
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Opérateurs génétiques
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate(self, individual: List[float]) -> float:
        """
        Évalue la performance d'un individu
        
        Args:
            individual: Liste des poids du modèle
            
        Returns:
            fitness: Score de performance
        """
        # Création d'un modèle avec les poids
        model = self.model_class()
        model.load_state_dict({
            name: torch.tensor(weight) 
            for name, weight in zip(model.state_dict().keys(), individual)
        })
        
        # Évaluation (à adapter selon le cas d'usage)
        # Exemple : précision + vitesse - taille
        return model.get_model_size() * 0.3 + model.evaluate_performance() * 0.7
    
    def optimize(self, generations: int = 100, cx_prob: float = 0.5, mut_prob: float = 0.2) -> BaseNanoModel:
        """
        Lance l'optimisation darwinienne
        
        Args:
            generations: Nombre de générations
            cx_prob: Probabilité de croisement
            mut_prob: Probabilité de mutation
            
        Returns:
            best_model: Meilleur modèle trouvé
        """
        # Création de la population initiale
        population = self.toolbox.population(n=self.population_size)
        
        # Évaluation initiale
        fitnesses = list(map(self.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = (fit,)
        
        # Évolution
        for gen in range(generations):
            # Sélection
            offspring = self.toolbox.select(population, len(population))
            offspring = list(map(self.toolbox.clone, offspring))
            
            # Croisement
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cx_prob:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Mutation
            for mutant in offspring:
                if np.random.random() < mut_prob:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            
            # Évaluation
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,)
            
            # Remplacement
            population[:] = offspring
            
            # Meilleur individu
            best_ind = tools.selBest(population, 1)[0]
            print(f"Génér. {gen}: Fitness = {best_ind.fitness.values[0]}")
        
        # Création du meilleur modèle
        best_model = self.model_class()
        best_model.load_state_dict({
            name: torch.tensor(weight) 
            for name, weight in zip(best_model.state_dict().keys(), best_ind)
        })
        
        return best_model
