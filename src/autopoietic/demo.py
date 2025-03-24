import numpy as np
from nano_autocoder import NanoAutoCoder
from force_field_agent import DigitalPheromoneSwarm

def main():
    """
    Démonstration de l'Auto-Réparation et de l'Essaim de Force.

    Cette fonction montre comment utiliser l'Auto-Réparation et l'Essaim de Force
    pour générer du code Python et prendre des décisions collectives.

    1. Démonstration de l'Auto-Réparation:
       - Crée une instance de l'Auto-Réparation
       - Génère du code Python pour une tâche spécifique
       - Vérifie que le code est fonctionnel

    2. Démonstration de l'Essaim de Force:
       - Crée une instance de l'Essaim de Force
       - Ajoute des agents
       - Simule des données d'entrée
       - Prend une décision collective
       - Vérifie l'état de l'essaim

    """
    # Création de l'autocoder
    print("\n1. Démonstration de l'Auto-Réparation")
    autocoder = NanoAutoCoder()
    
    # Test de la génération de code
    task = "calculate square root of a number"
    test_cases = [((4,), 2.0), ((9,), 3.0), ((16,), 4.0)]
    
    print(f"\nTâche: {task}")
    working_code = autocoder.self_repair(task, test_cases)
    print(f"\nCode généré:")
    print(working_code)
    
    # Création de l'essaim
    print("\n2. Démonstration de l'Essaim de Force")
    swarm = DigitalPheromoneSwarm(max_agents=100)
    
    # Ajout d'agents
    for _ in range(50):  # Création de 50 agents
        swarm.add_agent()
    
    # Simulation de données
    input_data = np.random.rand(10).tolist()
    
    print("\nDonnées d'entrée:", input_data)
    
    # Décision collective
    decision = swarm.collective_decision(input_data)
    print(f"\nDécision de l'essaim: {'Accept' if decision else 'Reject'}")
    
    # Statut de l'essaim
    print("\nÉtat de l'essaim:")
    status = swarm.get_swarm_status()
    for key, value in status.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
