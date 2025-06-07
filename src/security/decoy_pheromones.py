from typing import Dict, Any, Optional
import random
import time

# Attempt to import PheromoneNetwork, making it optional for standalone testing
try:
    from src.pheromones.digital_pheromones import PheromoneNetwork, DigitalPheromone
except ImportError:
    print("Warning: PheromoneNetwork not found. DecoyPheromoneSystem may have limited functionality.")
    # Define dummy classes if the import fails, to allow basic script execution
    class PheromoneNetwork:
        def __init__(self, db_path: str = "dummy_pheromones.db"):
            print(f"Dummy PheromoneNetwork initialized with db: {db_path}")
        def broadcast_pheromone(self, agent_id: str, signal_type: str, data: Dict[str, Any], ttl: int = 3600):
            print(f"Dummy broadcast: Agent {agent_id}, Type {signal_type}, Data {data}, TTL {ttl}")

    class DigitalPheromone: # Not directly used by DecoyPheromoneSystem but good for consistency
        pass


class DecoyPheromoneSystem:
    """
    Manages the generation and deployment of decoy pheromones and false traffic
    patterns to mislead and identify attackers. Part of the "Phéromones Leurres"
    concept in a revolutionary security architecture.
    """

    def __init__(self, pheromone_network: Optional[PheromoneNetwork] = None, agent_id_prefix: str = "decoy_agent_"):
        """
        Initializes the DecoyPheromoneSystem.

        Args:
            pheromone_network: An instance of PheromoneNetwork to interact with the actual
                               pheromone system. If None, a dummy one might be used or
                               functionality will be limited.
            agent_id_prefix: Prefix for generating IDs for decoy agents.
        """
        self.pheromone_network = pheromone_network if pheromone_network else PheromoneNetwork()
        self.agent_id_prefix = agent_id_prefix
        self.active_decoys: Dict[str, Any] = {}
        print("DecoyPheromoneSystem initialized.")

    def _generate_decoy_agent_id(self) -> str:
        """Generates a unique ID for a decoy agent."""
        return f"{self.agent_id_prefix}{random.randint(1000, 9999)}_{int(time.time() * 1000)}"

    def generate_false_traffic_patterns(self, duration_seconds: int = 60, intensity: int = 5) -> None:
        """
        Generates plausible but fictitious pheromone traffic to create noise
        and mislead observers or automated analysis systems.

        Args:
            duration_seconds: How long to generate false traffic.
            intensity: Number of false pheromones to generate per second (approx).
        """
        print(f"Generating false traffic patterns for {duration_seconds} seconds with intensity {intensity}...")
        end_time = time.time() + duration_seconds
        while time.time() < end_time:
            for _ in range(intensity):
                if time.time() >= end_time: break

                decoy_agent_id = self._generate_decoy_agent_id()
                signal_type = random.choice(["status_update", "resource_query", "data_fragment", "heartbeat"])
                data_payload = {
                    "timestamp": time.time(),
                    "source_ip": f"192.168.{random.randint(1,254)}.{random.randint(1,254)}", # Fake IP
                    "value": random.random() * 100,
                    "message": f"Decoy message {random.randint(1,1000)}"
                }
                ttl = random.randint(60, 600) # Short to medium TTL for decoys

                self.pheromone_network.broadcast_pheromone(
                    agent_id=decoy_agent_id,
                    signal_type=signal_type,
                    data=data_payload,
                    ttl=ttl
                )
            time.sleep(1) # Sleep for a second before next burst
        print("False traffic generation complete.")

    def deploy_distributed_honeypot_signals(self, number_of_decoys: int) -> None:
        """
        Deploys specific pheromone signals that act as honeypots.
        Interaction with these signals could indicate malicious activity.

        Args:
            number_of_decoys: The number of honeypot signals to deploy.
        """
        print(f"Deploying {number_of_decoys} distributed honeypot signals...")
        for i in range(number_of_decoys):
            decoy_agent_id = self._generate_decoy_agent_id() + f"_honeypot_{i}"
            honeypot_data = {
                "service_name": f"vulnerable_service_v{random.uniform(1,3):.1f}",
                "access_level": "simulated_admin_access",
                "description": "This is a honeypot signal. Interaction is logged.",
                "deployed_at": time.time()
            }
            # Use a specific signal type for honeypots for easier monitoring
            signal_type = "honeypot_beacon"
            ttl = 3600 * 24 # Longer TTL for persistent honeypots

            self.pheromone_network.broadcast_pheromone(
                agent_id=decoy_agent_id,
                signal_type=signal_type,
                data=honeypot_data,
                ttl=ttl
            )
            self.active_decoys[decoy_agent_id] = {"type": "honeypot", "data": honeypot_data, "signal_type": signal_type}
            print(f"Deployed honeypot signal from {decoy_agent_id}")
        print(f"{number_of_decoys} honeypot signals deployed.")

    def coordinate_disinformation(self, target_profile: Dict[str, Any], campaign_id: str) -> None:
        """
        Coordinates the spread of specific disinformation through pheromones,
        tailored to a target profile (e.g., an attacker's expected interests).

        Args:
            target_profile: A dictionary describing the target, e.g.,
                            {"interest": "financial_data", "sophistication": "medium"}
            campaign_id: An identifier for this disinformation campaign.
        """
        print(f"Coordinating disinformation campaign '{campaign_id}' for target profile: {target_profile}")

        # Example: Generate misleading "financial_data_location" pheromones
        if target_profile.get("interest") == "financial_data":
            for i in range(random.randint(3,7)): # Number of disinformation signals
                decoy_agent_id = self._generate_decoy_agent_id() + f"_disinfo_{campaign_id}_{i}"
                disinfo_payload = {
                    "data_type": "financial_projection_q3",
                    "location_hint": f"//fakeserver.local/share/quarterly_reports_archive_{random.randint(2018,2022)}/",
                    "credibility_score": random.uniform(0.7, 0.95), # Make it look credible
                    "access_key_hint": f"access_key_part_{random.choice(['alpha', 'beta', 'gamma'])}",
                    "campaign": campaign_id
                }
                signal_type = "info_leak_decoy" # Specific type for this kind of decoy
                ttl = random.randint(1800, 7200) # Medium TTL

                self.pheromone_network.broadcast_pheromone(
                    agent_id=decoy_agent_id,
                    signal_type=signal_type,
                    data=disinfo_payload,
                    ttl=ttl
                )
                self.active_decoys[decoy_agent_id] = {"type": "disinformation", "data": disinfo_payload, "signal_type": signal_type}
                print(f"Disinformation signal deployed by {decoy_agent_id} for campaign '{campaign_id}'")

        print(f"Disinformation campaign '{campaign_id}' deployment complete.")

    def check_honeypot_interactions(self) -> list:
        """
        (Conceptual) Checks if any of the deployed honeypot signals have been interacted with.
        In a real system, this would require monitoring specific pheromone responses or
        network traffic directed at decoy services.
        """
        # This is highly conceptual as it requires an external monitoring mechanism.
        # For now, it just returns a list of active honeypots.
        print("Checking honeypot interactions (conceptual)...")
        interacted_honeypots = []
        for agent_id, decoy_info in self.active_decoys.items():
            if decoy_info["type"] == "honeypot":
                # Simulate a random chance of interaction for demo purposes
                if random.random() < 0.1:
                    interacted_honeypots.append({
                        "honeypot_id": agent_id,
                        "interaction_time": time.time(),
                        "suspected_source": f"10.0.0.{random.randint(1,254)}" # Fake attacker IP
                    })
        if interacted_honeypots:
            print(f"Detected potential interactions with honeypots: {interacted_honeypots}")
        return interacted_honeypots


if __name__ == '__main__':
    # Example usage:
    # Initialize with a real PheromoneNetwork if available, otherwise it uses a dummy
    # from src.pheromones.digital_pheromones import PheromoneNetwork
    # p_network = PheromoneNetwork(db_path="main_pheromones.db")
    # decoy_system = DecoyPheromoneSystem(pheromone_network=p_network)

    # For standalone demo, it uses the dummy PheromoneNetwork
    decoy_system = DecoyPheromoneSystem()

    print("\n--- Generating False Traffic ---")
    decoy_system.generate_false_traffic_patterns(duration_seconds=3, intensity=2)

    print("\n--- Deploying Honeypot Signals ---")
    decoy_system.deploy_distributed_honeypot_signals(number_of_decoys=2)

    print("\n--- Coordinating Disinformation ---")
    profile = {"interest": "financial_data", "sophistication": "medium"}
    decoy_system.coordinate_disinformation(target_profile=profile, campaign_id="q3_earnings_leak")

    print("\n--- Checking Honeypot Interactions (Conceptual) ---")
    interactions = decoy_system.check_honeypot_interactions()
    if interactions:
        print(f"Detected {len(interactions)} honeypot interaction(s).")
    else:
        print("No honeypot interactions detected in this conceptual check.")

    print(f"Total active decoys: {len(decoy_system.active_decoys)}")
