import threading
import random
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import numpy as np

class ForceFieldAgent:
    """Agent de l'essaim utilisant des modèles minimalistes"""
    
    def __init__(self, agent_id: int, swarm: 'DigitalPheromoneSwarm', model_size: int = 10):
        """
        Args:
            agent_id: Identifiant unique de l'agent
            swarm: Référence à la colonie
            model_size: Taille maximale du modèle en Ko
        """
        self.id = agent_id
        self.swarm = swarm
        self.model_size = model_size
        self.local_model = self._initialize_model()
        
    def _initialize_model(self) -> Dict[str, Any]:
        """Initialise un modèle minimaliste"""
        # Modèle simple : perceptron avec poids aléatoires
        return {
            'weights': np.random.uniform(-1, 1, 10),
            'bias': np.random.uniform(-1, 1)
        }
    
    def process(self, data: List[float]) -> float:
        """
        Traite les données via le modèle local
        
        Args:
            data: Données d'entrée
            
        Returns:
            prediction: Prédiction normalisée
        """
        try:
            # Vérification de la taille
            if len(data) > 100:
                raise ValueError("Données trop grandes")
            
            # Inférence
            activation = np.dot(self.local_model['weights'], data) + self.local_model['bias']
            prediction = self._activation_function(activation)
            
            # Partage de connaissance
            self.swarm.leave_pheromone(self.id, prediction)
            
            return prediction
        except Exception as e:
            print(f"Erreur de traitement: {str(e)}")
            return 0.0
    
    def _activation_function(self, x: float) -> float:
        """Fonction d'activation sigmoïde"""
        return 1 / (1 + np.exp(-x))
    
    def update_model(self, new_weights: np.ndarray, new_bias: float) -> None:
        """Met à jour le modèle local"""
        if len(new_weights) != len(self.local_model['weights']):
            raise ValueError("Taille des poids invalide")
            
        # Mise à jour avec lissage
        alpha = 0.1  # Facteur de lissage
        self.local_model['weights'] = (1 - alpha) * self.local_model['weights'] + alpha * new_weights
        self.local_model['bias'] = (1 - alpha) * self.local_model['bias'] + alpha * new_bias

class DigitalPheromoneSwarm:
    """Essaim d'agents collaboratifs utilisant des phéromones numériques"""
    
    def __init__(self, max_agents: int = 1000):
        """
        Args:
            max_agents: Nombre maximum d'agents dans l'essaim
        """
        self.max_agents = max_agents
        self.agents = {}
        self.pheromones = defaultdict(list)
        self.lock = threading.Lock()
        self._initialize_communication_protocol()
    
    def _initialize_communication_protocol(self) -> None:
        """Initialise le protocole de communication"""
        self.communication_protocol = {
            'pheromone_threshold': 0.1,  # Seuil pour la propagation
            'evaporation_rate': 0.05,    # Taux d'évaporation
            'max_history': 100           # Taille maximale de l'historique
        }
    
    def add_agent(self) -> int:
        """Ajoute un nouvel agent à l'essaim"""
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Limite d'agents atteinte ({self.max_agents})")
            
        agent_id = len(self.agents) + 1
        self.agents[agent_id] = ForceFieldAgent(agent_id, self)
        return agent_id
    
    def leave_pheromone(self, agent_id: int, value: float) -> None:
        """Laisse une phéromone numérique"""
        with self.lock:
            self.pheromones[agent_id].append(value)
            
            # Évaporation des phéromones
            if len(self.pheromones[agent_id]) > self.communication_protocol['max_history']:
                self.pheromones[agent_id].pop(0)
            
            # Propagation des phéromones
            self._propagate_pheromones(agent_id, value)
    
    def _propagate_pheromones(self, source_id: int, value: float) -> None:
        """Propage les phéromones dans l'essaim"""
        # Sélection aléatoire des voisins
        neighbors = random.sample(
            list(self.agents.keys()),
            min(5, len(self.agents))  # Maximum 5 voisins
        )
        
        for neighbor_id in neighbors:
            if neighbor_id != source_id:
                # Propagation avec décroissance
                decayed_value = value * (1 - self.communication_protocol['evaporation_rate'])
                self.pheromones[neighbor_id].append(decayed_value)
    
    def collective_decision(self, input_data: List[float]) -> bool:
        """
        Prend une décision collective via l'essaim
        
        Args:
            input_data: Données d'entrée pour l'essaim
            
        Returns:
            decision: Décision collective (True/False)
        """
        # Exécution parallèle
        threads = []
        results = []
        
        def agent_task(agent: ForceFieldAgent):
            res = agent.process(input_data)
            results.append(res)
        
        for agent in self.agents.values():
            thread = threading.Thread(target=agent_task, args=(agent,))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # Décision démocratique
        avg_prediction = np.mean(results)
        return avg_prediction > 0.5
    
    def get_swarm_status(self) -> Dict[str, Any]:
        """Retourne l'état de l'essaim"""
        return {
            'num_agents': len(self.agents),
            'pheromone_count': sum(len(p) for p in self.pheromones.values()),
            'communication_protocol': self.communication_protocol
        }
