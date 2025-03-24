import numpy as np
from core import BigBangCore
from agent import Agent
import time

def simulate_ambient_signals():
    """Simule des signaux ambients"""
    return {
        'vibration': np.random.normal(0, 1, 100),
        'light': np.random.uniform(0, 100),
        'sound': np.random.normal(0, 1, 1000)
    }

def main():
    # Création du noyau initial
    core = BigBangCore()
    
    # Simulation sur 10 cycles
    for cycle in range(10):
        print(f"\nCycle {cycle + 1}")
        
        # Simulation des signaux ambients
        ambient_signals = simulate_ambient_signals()
        
        # Apprentissage
        core.learn(ambient_signals)
        
        # Compression
        core.compress()
        
        # Création d'un nouvel agent
        if cycle % 2 == 0:  # Nouvel agent tous les 2 cycles
            agent = core.spawn_agent()
            print(f"Agent créé: {agent.get_status()}")
        
        # Statut du noyau
        status = core.get_status()
        print(f"Taille: {status['size']} octets")
        print(f"Énergie: {status['energy']:.2f}")
        print(f"Tâches: {status['num_tasks']}")
        print(f"Agents: {status['num_agents']}")
        
        # Pause entre les cycles
        time.sleep(1)

if __name__ == "__main__":
    main()
