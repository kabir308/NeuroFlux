import sqlite3
from typing import Dict, Any
import time
from pathlib import Path

class DigitalPheromone:
    """Phéromone numérique persistante"""
    
    def __init__(self, 
                 agent_id: str, 
                 signal_type: str, 
                 data: Dict[str, Any],
                 ttl: int = 3600):  # TTL par défaut : 1 heure
        """
        Args:
            agent_id: ID de l'agent émetteur
            signal_type: Type de signal (ex: "alert", "knowledge", "coordination")
            data: Données associées
            ttl: Temps de vie en secondes
        """
        self.agent_id = agent_id
        self.signal_type = signal_type
        self.data = data
        self.ttl = ttl
        self.timestamp = int(time.time())
    
    def is_expired(self) -> bool:
        """Vérifie si la phéromone est expirée"""
        return (int(time.time()) - self.timestamp) > self.ttl

class PheromoneDatabase:
    """Base de données pour les phéromones numériques"""
    
    def __init__(self, db_path: str = "pheromones.db"):
        """
        Args:
            db_path: Chemin vers la base de données SQLite
        """
        self.db_path = Path(db_path)
        self._init_db()
    
    def _init_db(self):
        """Initialise la base de données"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS pheromones (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    agent_id TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    ttl INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL
                )
            ''')
            conn.commit()
    
    def add_pheromone(self, pheromone: DigitalPheromone) -> None:
        """
        Ajoute une phéromone à la base de données
        
        Args:
            pheromone: Phéromone numérique
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO pheromones (
                    agent_id, signal_type, data, ttl, timestamp
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                pheromone.agent_id,
                pheromone.signal_type,
                str(pheromone.data),
                pheromone.ttl,
                pheromone.timestamp
            ))
            conn.commit()
    
    def get_pheromones(self, 
                      signal_type: str = None, 
                      agent_id: str = None) -> list:
        """
        Récupère les phéromones non expirées
        
        Args:
            signal_type: Type de signal (optionnel)
            agent_id: ID de l'agent (optionnel)
            
        Returns:
            list: Liste des phéromones
        """
        query = '''
            SELECT * FROM pheromones 
            WHERE timestamp + ttl > ?
        '''
        params = [int(time.time())]
        
        if signal_type:
            query += " AND signal_type = ?"
            params.append(signal_type)
            
        if agent_id:
            query += " AND agent_id = ?"
            params.append(agent_id)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
        return [
            DigitalPheromone(
                agent_id=row[1],
                signal_type=row[2],
                data=eval(row[3]),
                ttl=row[4],
                timestamp=row[5]
            )
            for row in rows
        ]
    
    def clean_expired(self) -> None:
        """Supprime les phéromones expirées"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM pheromones 
                WHERE timestamp + ttl < ?
            ''', [int(time.time())])
            conn.commit()

class PheromoneNetwork:
    def __init__(self, db_path: str = "pheromones.db"):
        """
        Manages the network of digital pheromones and collective behaviors.

        Args:
            db_path: Path to the pheromone database.
        """
        self.db = PheromoneDatabase(db_path)

    def broadcast_pheromone(self, agent_id: str, signal_type: str, data: Dict[str, Any], ttl: int = 3600):
        """
        Broadcasts a pheromone to the network.
        This involves creating a DigitalPheromone and adding it to the database.
        Future extensions could involve multi-protocol dispatch to other devices.

        Args:
            agent_id: ID of the agent emitting the pheromone.
            signal_type: Type of signal (e.g., "alert", "knowledge", "coordination").
            data: Data payload of the pheromone.
            ttl: Time-to-live for the pheromone in seconds.
        """
        pheromone = DigitalPheromone(agent_id=agent_id, signal_type=signal_type, data=data, ttl=ttl)
        self.db.add_pheromone(pheromone)
        # TODO: Implement multi-protocol dispatch for actual network broadcast
        print(f"Pheromone broadcasted: Agent {agent_id}, Type {signal_type}")

    def collective_decision(self, proposal: Any, required_consensus: float = 0.51) -> bool:
        """
        Initiates a collective decision-making process based on a proposal.
        This is a placeholder for a distributed consensus mechanism.

        Args:
            proposal: The proposal to be decided upon.
            required_consensus: The fraction of positive responses needed for approval.

        Returns:
            True if the proposal is accepted by consensus, False otherwise.
        """
        # TODO: Implement distributed consensus mechanism.
        # This might involve broadcasting a "vote_request" pheromone,
        # collecting "vote_response" pheromones, and tallying results.
        print(f"Collective decision process initiated for proposal: {proposal}")
        # Placeholder: Simulate a positive outcome
        return True

    def auto_repair_trigger(self, anomaly_type: str, anomaly_data: Dict[str, Any]):
        """
        Triggers an auto-repair mechanism in response to a detected anomaly.
        This typically involves broadcasting a specific type of pheromone
        to alert relevant systems or agents.

        Args:
            anomaly_type: The type of anomaly detected (e.g., "node_failure", "performance_degradation").
            anomaly_data: Data associated with the anomaly.
        """
        repair_signal_data = {
            "anomaly_type": anomaly_type,
            "details": anomaly_data,
            "timestamp": time.time()
        }
        # Broadcast a high-priority alert pheromone for auto-repair
        self.broadcast_pheromone(
            agent_id="PheromoneNetworkSystem",
            signal_type="auto_repair_alert",
            data=repair_signal_data,
            ttl=7200 # Longer TTL for important alerts
        )
        print(f"Auto-repair sequence triggered for anomaly: {anomaly_type}")

def main():
    # Exemple d'utilisation
    db = PheromoneDatabase()
    
    # Créer une phéromone
    pheromone = DigitalPheromone(
        agent_id="agent_1",
        signal_type="knowledge",
        data={"topic": "AI", "confidence": 0.9},
        ttl=3600
    )
    
    # Ajouter la phéromone
    db.add_pheromone(pheromone)
    
    # Récupérer les phéromones
    pheromones = db.get_pheromones()
    print("\nPhéromones actives:")
    for p in pheromones:
        print(f"- Agent: {p.agent_id}, Type: {p.signal_type}")

if __name__ == "__main__":
    main()
