import hashlib
import json
import time
from typing import List, Dict, Any, Optional

class Block:
    def __init__(self, index: int, timestamp: float, transactions: List[Dict[str, Any]], previous_hash: str, proof: int = 0):
        self.index = index
        self.timestamp = timestamp
        self.transactions = transactions
        self.previous_hash = previous_hash
        self.proof = proof
        self.hash = self.calculate_hash()

    def calculate_hash(self) -> str:
        """Calculate SHA-256 hash of the block content."""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": self.timestamp,
            "transactions": self.transactions,
            "previous_hash": self.previous_hash,
            "proof": self.proof
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "timestamp_formatted": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp)),
            "transactions": self.transactions,
            "tx_count": len(self.transactions),
            "previous_hash": self.previous_hash,
            "proof": self.proof,
            "hash": self.hash
        }


class Blockchain:
    def __init__(self, difficulty: int = 2):
        self.chain: List[Block] = []
        self.pending_transactions: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """Create initial genesis block #0."""
        genesis_tx = [{
            "tx_id": "GENESIS_TX_0000000000000000",
            "timestamp": time.time(),
            "amount": 0.0,
            "fraud_probability": 0.0,
            "is_fraud": False,
            "risk_level": "SYSTEM",
            "note": "Genesis Block - Credit Card Fraud Prevention Ledger Initialized"
        }]
        genesis_block = Block(
            index=0,
            timestamp=time.time(),
            transactions=genesis_tx,
            previous_hash="0" * 64,
            proof=100
        )
        # Mine genesis block hash to ensure it fulfills proof of work if required, or calculate directly
        genesis_block.hash = genesis_block.calculate_hash()
        self.chain.append(genesis_block)

    def get_last_block(self) -> Block:
        return self.chain[-1]

    def add_transaction(self, amount: float, fraud_probability: float, is_fraud: bool, risk_level: str, hour: Optional[int] = None, feature_summary: Optional[str] = None) -> Dict[str, Any]:
        """Add a fraud assessment transaction to pending pool and auto-assign cryptographic transaction hash."""
        tx_time = time.time()
        tx_payload = {
            "tx_id": f"TX_{int(tx_time*1000)}_{len(self.pending_transactions)+1}",
            "timestamp": tx_time,
            "amount": amount,
            "hour": hour,
            "fraud_probability": round(fraud_probability, 4),
            "fraud_percentage": round(fraud_probability * 100, 2),
            "is_fraud": is_fraud,
            "risk_level": risk_level,
            "feature_summary": feature_summary or "Standard ML Feature Vector"
        }
        
        # Calculate SHA-256 tx hash
        tx_json = json.dumps(tx_payload, sort_keys=True)
        tx_payload["tx_hash"] = hashlib.sha256(tx_json.encode('utf-8')).hexdigest()
        
        self.pending_transactions.append(tx_payload)
        
        # Auto-mine if pending transactions reach 3 items so blocks populate automatically during testing
        mined_block = None
        if len(self.pending_transactions) >= 3:
            mined_block = self.mine_pending_transactions()
            
        return {
            "transaction": tx_payload,
            "pending_count": len(self.pending_transactions),
            "mined_block": mined_block.to_dict() if mined_block else None
        }

    def proof_of_work(self, block: Block) -> int:
        """Proof of Work algorithm: find nonce such that hash starts with N zeros."""
        block.proof = 0
        computed_hash = block.calculate_hash()
        target_prefix = "0" * self.difficulty
        
        while not computed_hash.startswith(target_prefix):
            block.proof += 1
            computed_hash = block.calculate_hash()
            
        return block.proof

    def mine_pending_transactions(self, miner_address: str = "SYSTEM_AI_ENGINE") -> Block:
        """Mine pending transactions into a new Block and append to chain."""
        if not self.pending_transactions:
            # Add a heart-beat idle record if manual mine triggered with empty pool
            self.add_transaction(amount=0.0, fraud_probability=0.0, is_fraud=False, risk_level="LOW", feature_summary="Manual Block Mining Verification")

        last_block = self.get_last_block()
        new_block = Block(
            index=len(self.chain),
            timestamp=time.time(),
            transactions=list(self.pending_transactions),
            previous_hash=last_block.hash
        )
        
        # Run Proof of Work
        self.proof_of_work(new_block)
        new_block.hash = new_block.calculate_hash()
        
        # Append to chain and clear pending transactions
        self.chain.append(new_block)
        self.pending_transactions = []
        return new_block

    def validate_chain(self) -> Dict[str, Any]:
        """Validate SHA-256 hashes and block linkages across the entire blockchain ledger."""
        target_prefix = "0" * self.difficulty
        
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # 1. Verify previous hash pointer
            if current.previous_hash != previous.hash:
                return {
                    "is_valid": False,
                    "error_type": "PREVIOUS_HASH_MISMATCH",
                    "error_block_index": current.index,
                    "message": f"Block #{current.index} previous_hash link is broken! Expected '{previous.hash[:16]}...', got '{current.previous_hash[:16]}...'."
                }
                
            # 2. Re-calculate current hash
            recalculated_hash = current.calculate_hash()
            if current.hash != recalculated_hash:
                return {
                    "is_valid": False,
                    "error_type": "HASH_CORRUPTED",
                    "error_block_index": current.index,
                    "message": f"Block #{current.index} data has been tampered! Stored hash '{current.hash[:16]}...' does not match calculated hash '{recalculated_hash[:16]}...'."
                }

            # 3. Verify Proof of Work
            if not current.hash.startswith(target_prefix):
                return {
                    "is_valid": False,
                    "error_type": "POW_INVALID",
                    "error_block_index": current.index,
                    "message": f"Block #{current.index} does not satisfy Proof-of-Work difficulty '{target_prefix}'."
                }

        return {
            "is_valid": True,
            "error_type": None,
            "error_block_index": None,
            "message": "Blockchain integrity verified. All cryptographic hashes and Proof-of-Work links are 100% valid."
        }

    def simulate_tamper(self, block_index: int, transaction_index: int = 0, new_amount: float = 0.0, flip_fraud: bool = True) -> Dict[str, Any]:
        """Tamper with a block's transaction data to demonstrate cryptographic break detection."""
        if block_index <= 0 or block_index >= len(self.chain):
            return {"success": False, "message": f"Invalid block index {block_index}. Cannot tamper Genesis Block #0."}

        target_block = self.chain[block_index]
        if not target_block.transactions or transaction_index >= len(target_block.transactions):
            return {"success": False, "message": f"No transaction found at index {transaction_index} in Block #{block_index}."}

        # Modify transaction data behind the scenes without updating block hash or proof
        tx = target_block.transactions[transaction_index]
        original_amount = tx["amount"]
        original_is_fraud = tx["is_fraud"]

        tx["amount"] = new_amount
        if flip_fraud:
            tx["is_fraud"] = not original_is_fraud
            tx["fraud_probability"] = 0.01 if not original_is_fraud else 0.99
            tx["risk_level"] = "LOW (TAMPERED)" if not original_is_fraud else "CRITICAL (TAMPERED)"
            tx["note"] = "⚠️ TAMPERED DATA DETECTED"

        return {
            "success": True,
            "tampered_block_index": block_index,
            "original_amount": original_amount,
            "new_amount": new_amount,
            "original_is_fraud": original_is_fraud,
            "new_is_fraud": tx["is_fraud"],
            "message": f"Block #{block_index} transaction data tampered successfully. Run ledger validation to see detection!"
        }

    def repair_chain(self) -> Dict[str, Any]:
        """Re-calculate hashes and re-mine PoW for tampered blocks to repair chain validity."""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            current.previous_hash = previous.hash
            self.proof_of_work(current)
            current.hash = current.calculate_hash()

        return {
            "success": True,
            "message": "Blockchain hashes and Proof-of-Work nonces successfully recalculated. Ledger state restored to VALID."
        }

    def get_chain_data(self) -> Dict[str, Any]:
        validation = self.validate_chain()
        return {
            "chain": [b.to_dict() for b in self.chain],
            "length": len(self.chain),
            "pending_transactions": self.pending_transactions,
            "pending_count": len(self.pending_transactions),
            "difficulty": self.difficulty,
            "validation": validation,
            "total_transactions": sum(len(b.transactions) for b in self.chain) + len(self.pending_transactions)
        }
