from __future__ import annotations # For type hinting DigitalMitosisAgent in its own methods
from typing import Dict, Any, List, Optional
import random
import time
import uuid # For unique agent IDs

class DigitalMitosisAgent:
    """
    Represents an AI agent capable of "Digital Mitosis" - a form of
    self-replication, selective component inheritance, and fragmentation
    for survival and evolution, as part of Bio-Mimetic Advanced concepts.
    """

    def __init__(self, agent_id: Optional[str] = None, dna: Optional[Dict[str, Any]] = None, generation: int = 0):
        """
        Initializes a DigitalMitosisAgent.

        Args:
            agent_id: A unique identifier for the agent. If None, a new one is generated.
            dna: A dictionary representing the agent's "genetic" makeup,
                 containing its components, parameters, and capabilities.
            generation: The generation number of this agent.
        """
        self.agent_id: str = agent_id or str(uuid.uuid4())
        self.generation: int = generation

        # Default DNA structure if none provided
        default_dna = {
            "core_logic_module": {"version": random.uniform(1.0, 3.0), "complexity": random.randint(1,10)},
            "learning_algorithm": {"type": random.choice(["q_learning", "genetic_algorithm", "neural_network"]), "fitness": random.random()},
            "communication_protocol": {"type": "pheromone_v2", "efficiency": random.random()},
            "resource_requirements": {"cpu": random.randint(10,100), "memory": random.randint(32,512)},
            "age": 0, # Represents lifetime or operational cycles
            "health": 1.0 # Normalized health score
        }
        self.dna: Dict[str, Any] = dna if dna is not None else default_dna

        print(f"DigitalMitosisAgent {self.agent_id} (Gen {self.generation}) initialized. DNA: {self.dna}")

    def _mutate_dna(self, original_dna: Dict[str, Any], mutation_rate: float = 0.05) -> Dict[str, Any]:
        """Simulates mutation during replication."""
        mutated_dna = original_dna.copy() # Start with a copy
        for key, value in mutated_dna.items():
            if isinstance(value, dict): # Mutate sub-components
                mutated_dna[key] = self._mutate_dna(value, mutation_rate)
            elif isinstance(value, (int, float)) and random.random() < mutation_rate:
                if isinstance(value, int):
                    mutated_dna[key] += random.randint(-max(1, value // 10), max(1, value // 10))
                else: # float
                    mutated_dna[key] += random.uniform(-value * 0.1, value * 0.1)
            # Could add more mutation types (e.g., for string types like algorithm types)
        return mutated_dna

    def check_survival_conditions(self, environment_factors: Dict[str, Any]) -> bool:
        """
        Checks if the current conditions warrant survival or if actions like
        replication or fragmentation are necessary.

        Args:
            environment_factors: A dictionary describing the current environment,
                                 e.g., {"resource_availability": 0.3, "threat_level": 0.8}

        Returns:
            True if survival conditions are met and no drastic action is needed,
            False otherwise (suggesting replication or fragmentation might be beneficial).
        """
        # Placeholder logic
        resource_ok = environment_factors.get("resource_availability", 1.0) >= (self.dna["resource_requirements"]["cpu"]/100 + self.dna["resource_requirements"]["memory"]/512)/2
        threat_low = environment_factors.get("threat_level", 0.0) < 0.7
        health_ok = self.dna.get("health", 1.0) > 0.5

        if resource_ok and threat_low and health_ok:
            print(f"Agent {self.agent_id}: Survival conditions met.")
            return True
        else:
            print(f"Agent {self.agent_id}: Survival conditions NOT met (ResourceOK: {resource_ok}, ThreatLow: {threat_low}, HealthOK: {health_ok}). Consider mitosis.")
            return False

    def initiate_replication(self, strategy: str = "standard_mitosis") -> DigitalMitosisAgent:
        """
        Creates a new agent (offspring) through replication.
        The offspring inherits DNA, possibly with mutations.

        Args:
            strategy: The replication strategy to use (e.g., "standard_mitosis",
                      "fission_optimized_for_learning"). Placeholder for now.

        Returns:
            A new DigitalMitosisAgent instance (the offspring).
        """
        print(f"Agent {self.agent_id} initiating replication (Strategy: {strategy})...")

        # DNA for offspring is a mutated copy of the parent's DNA
        offspring_dna = self._mutate_dna(self.dna)

        offspring_agent = DigitalMitosisAgent(dna=offspring_dna, generation=self.generation + 1)
        print(f"Agent {self.agent_id} produced offspring {offspring_agent.agent_id} (Gen {offspring_agent.generation}).")
        return offspring_agent

    def select_fittest_components(self) -> Dict[str, Any]:
        """
        Analyzes the agent's own DNA components and selects the "fittest" ones,
        perhaps for preferential replication or to form a specialized fragment.
        "Fitness" is determined by component-specific metrics (e.g., efficiency, accuracy).

        Returns:
            A dictionary containing the selected fittest components from the DNA.
        """
        print(f"Agent {self.agent_id} selecting fittest components...")
        fittest_components = {}

        # Placeholder: Select components with high "fitness" or "efficiency" scores
        # or low "complexity" / "resource_requirements"
        if self.dna["learning_algorithm"].get("fitness", 0) > 0.7:
            fittest_components["learning_algorithm"] = self.dna["learning_algorithm"]

        if self.dna["communication_protocol"].get("efficiency", 0) > 0.6:
            fittest_components["communication_protocol"] = self.dna["communication_protocol"]

        if self.dna["core_logic_module"].get("complexity", 10) < 5 and \
           self.dna["core_logic_module"].get("version", 0) > 1.5 :
             fittest_components["core_logic_module"] = self.dna["core_logic_module"]

        print(f"Agent {self.agent_id} selected components: {fittest_components if fittest_components else 'None met criteria'}")
        return fittest_components

    def fragment_for_survival(self, number_of_fragments: int = 2) -> List[DigitalMitosisAgent]:
        """
        Splits the agent into multiple, potentially specialized, fragments.
        Each fragment is a new agent with a subset or modified version of the original DNA.
        This is a strategy for survival in harsh conditions or for task distribution.

        Args:
            number_of_fragments: The number of fragments to create.

        Returns:
            A list of new DigitalMitosisAgent instances (the fragments).
        """
        print(f"Agent {self.agent_id} fragmenting into {number_of_fragments} parts for survival...")
        fragments: List[DigitalMitosisAgent] = []

        # Basic fragmentation: divide resources and potentially specialize components
        original_cpu_per_fragment = self.dna["resource_requirements"]["cpu"] / number_of_fragments
        original_mem_per_fragment = self.dna["resource_requirements"]["memory"] / number_of_fragments

        for i in range(number_of_fragments):
            fragment_dna = self.dna.copy() # Start with a copy of parent DNA

            # Mutate and specialize fragment DNA
            fragment_dna["resource_requirements"] = {"cpu": max(1, int(original_cpu_per_fragment * random.uniform(0.8, 1.2))),
                                                   "memory": max(1, int(original_mem_per_fragment * random.uniform(0.8, 1.2)))}
            fragment_dna["agent_type"] = f"fragment_specialized_{i}" # Mark as fragment

            # Example of specialization: one fragment focuses on learning, another on communication
            if i == 0 and "learning_algorithm" in fragment_dna: # First fragment enhances learning
                fragment_dna["learning_algorithm"]["fitness"] = min(1.0, fragment_dna["learning_algorithm"].get("fitness",0.5) * 1.2)
            elif i == 1 and "communication_protocol" in fragment_dna: # Second fragment enhances communication
                fragment_dna["communication_protocol"]["efficiency"] = min(1.0, fragment_dna["communication_protocol"].get("efficiency",0.5) * 1.2)

            fragment_dna = self._mutate_dna(fragment_dna, mutation_rate=0.1) # Higher mutation for fragments adapting

            fragment_agent = DigitalMitosisAgent(dna=fragment_dna, generation=self.generation + 1)
            fragments.append(fragment_agent)
            print(f"  Fragment {fragment_agent.agent_id} (Gen {fragment_agent.generation}) created.")

        # Optionally, the original agent might "terminate" after fragmentation
        # self.health = 0
        # print(f"Agent {self.agent_id} has terminated after fragmentation.")
        return fragments

    def age_and_update_health(self, cycles: int = 1, stress_factor: float = 0.01):
        """Simulates aging and health degradation over operational cycles."""
        self.dna["age"] = self.dna.get("age",0) + cycles
        # Health degrades based on age and stress
        self.dna["health"] = max(0, self.dna.get("health", 1.0) - (cycles * stress_factor) * (1 + self.dna.get("age",0)*0.01) )
        print(f"Agent {self.agent_id} aged. Current age: {self.dna['age']}, health: {self.dna['health']:.2f}")


if __name__ == '__main__':
    parent_agent = DigitalMitosisAgent(agent_id="parent_alpha")

    print("\n--- Simulating Agent Lifecycle ---")
    parent_agent.age_and_update_health(cycles=10, stress_factor=0.02)

    environment = {"resource_availability": 0.4, "threat_level": 0.6} # Moderately challenging
    if not parent_agent.check_survival_conditions(environment):
        print("\n--- Survival conditions not optimal, attempting replication ---")
        offspring = parent_agent.initiate_replication()
        offspring.age_and_update_health(cycles=5, stress_factor=0.01) # Offspring lives a bit

        print("\n--- Parent selecting fittest components (conceptual) ---")
        fittest = parent_agent.select_fittest_components()

        print("\n--- Parent attempting fragmentation due to continued stress ---")
        environment_harsh = {"resource_availability": 0.2, "threat_level": 0.85}
        parent_agent.dna["health"] = 0.3 # Simulate further health degradation
        if not parent_agent.check_survival_conditions(environment_harsh):
             fragments = parent_agent.fragment_for_survival(number_of_fragments=2)
             for frag in fragments:
                 frag.age_and_update_health(cycles=3)
    else:
        print(f"Agent {parent_agent.agent_id} continues normal operation.")
