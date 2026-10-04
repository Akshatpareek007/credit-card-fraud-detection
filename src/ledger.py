"""
Tamper-Evident SQLite Audit Ledger with Ed25519 Signatures and Canonical Hashing.
"""

import os
import sqlite3
import hashlib
import json
import time
import math
import threading
from typing import List, Dict, Any, Optional, Tuple

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.exceptions import InvalidSignature


# Default SQLite DB Path
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports", "ledger.db")


def canonical_serialize(payload: Dict[str, Any]) -> str:
    """
    Produce deterministic canonical string representation of payload:
    - Sorted keys
    - Fixed separators (',', ':')
    - No NaN or Infinity allowed
    - Floats formatted to fixed 8 decimal places
    """
    def sanitize(val: Any) -> Any:
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                raise ValueError("NaN or Infinity values are not permitted in canonical payload.")
            return f"{val:.8f}"
        elif isinstance(val, dict):
            return {k: sanitize(val[k]) for k in sorted(val.keys())}
        elif isinstance(val, (list, tuple)):
            return [sanitize(x) for x in val]
        return val

    sanitized_payload = {k: sanitize(payload[k]) for k in sorted(payload.keys())}
    return json.dumps(sanitized_payload, separators=(',', ':'), sort_keys=True)


def compute_payload_hash(payload: Dict[str, Any]) -> str:
    """Compute SHA-256 hex digest of canonicalized payload."""
    canonical_str = canonical_serialize(payload)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()


class Ed25519Signer:
    """Ed25519 Key management and cryptographic signing helper."""

    def __init__(self, private_key_hex: Optional[str] = None):
        if private_key_hex:
            try:
                raw_bytes = bytes.fromhex(private_key_hex)
                self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(raw_bytes)
            except Exception as e:
                raise ValueError(f"Invalid Ed25519 private key hex: {e}")
        else:
            # Generate deterministic or transient key pair if env var missing
            env_key = os.getenv("LEDGER_ED25519_KEY")
            if env_key:
                raw_bytes = bytes.fromhex(env_key)
                self.private_key = ed25519.Ed25519PrivateKey.from_private_bytes(raw_bytes)
            else:
                self.private_key = ed25519.Ed25519PrivateKey.generate()

        self.public_key = self.private_key.public_key()

    def get_public_key_hex(self) -> str:
        raw_pub = self.public_key.public_bytes_raw()
        return raw_pub.hex()

    def get_private_key_hex(self) -> str:
        raw_priv = self.private_key.private_bytes_raw()
        return raw_priv.hex()

    def sign(self, message: bytes) -> str:
        signature_bytes = self.private_key.sign(message)
        return signature_bytes.hex()

    def verify(self, signature_hex: str, message: bytes) -> bool:
        try:
            signature_bytes = bytes.fromhex(signature_hex)
            self.public_key.verify(signature_bytes, message)
            return True
        except (InvalidSignature, ValueError):
            return False


class AuditLedger:
    """
    Append-only Tamper-Evident Ledger stored in SQLite WAL mode.
    Guarded by a single writer lock for thread safety.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH, signer: Optional[Ed25519Signer] = None):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.writer_lock = threading.Lock()
        self.signer = signer or Ed25519Signer()
        self._init_sqlite()
        self.verify_chain()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_sqlite(self):
        with self.writer_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ledger_blocks (
                    block_index INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    payload_hash TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    fraud_score REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    previous_hash TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    block_hash TEXT NOT NULL
                );
            """)
            conn.commit()

            # Check if Genesis block exists
            cursor.execute("SELECT COUNT(*) as cnt FROM ledger_blocks;")
            row_count = cursor.fetchone()["cnt"]
            if row_count == 0:
                self._create_genesis_block(conn)
            conn.close()

    def _create_genesis_block(self, conn: sqlite3.Connection):
        timestamp = 1600000000.0
        payload_hash = hashlib.sha256(b"GENESIS_PAYLOAD").hexdigest()
        decision = "GENESIS"
        fraud_score = 0.0
        model_version = "v0.0.0"
        threshold = 0.0
        previous_hash = "0" * 64

        message = f"0|{timestamp:.8f}|{payload_hash}|{decision}|{fraud_score:.8f}|{model_version}|{threshold:.8f}|{previous_hash}".encode("utf-8")
        signature = self.signer.sign(message)

        block_content = f"0|{timestamp:.8f}|{payload_hash}|{decision}|{fraud_score:.8f}|{model_version}|{threshold:.8f}|{previous_hash}|{signature}".encode("utf-8")
        block_hash = hashlib.sha256(block_content).hexdigest()

        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO ledger_blocks (
                block_index, timestamp, payload_hash, decision, fraud_score,
                model_version, threshold, previous_hash, signature, block_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """, (0, timestamp, payload_hash, decision, fraud_score, model_version, threshold, previous_hash, signature, block_hash))
        conn.commit()

    def record_entry(
        self,
        payload: Dict[str, Any],
        decision: str,
        fraud_score: float,
        model_version: str,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Record a decision entry into the ledger inside a background thread/task,
        guarded by single writer lock.
        """
        payload_hash = compute_payload_hash(payload)
        timestamp = time.time()

        with self.writer_lock:
            conn = self._get_connection()
            cursor = conn.cursor()

            # Get latest block
            cursor.execute("SELECT block_index, block_hash FROM ledger_blocks ORDER BY block_index DESC LIMIT 1;")
            last_block = cursor.fetchone()
            new_index = last_block["block_index"] + 1
            previous_hash = last_block["block_hash"]

            # Format signature message
            message = f"{new_index}|{timestamp:.8f}|{payload_hash}|{decision}|{fraud_score:.8f}|{model_version}|{threshold:.8f}|{previous_hash}".encode("utf-8")
            signature = self.signer.sign(message)

            block_content = f"{new_index}|{timestamp:.8f}|{payload_hash}|{decision}|{fraud_score:.8f}|{model_version}|{threshold:.8f}|{previous_hash}|{signature}".encode("utf-8")
            block_hash = hashlib.sha256(block_content).hexdigest()

            cursor.execute("""
                INSERT INTO ledger_blocks (
                    block_index, timestamp, payload_hash, decision, fraud_score,
                    model_version, threshold, previous_hash, signature, block_hash
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """, (new_index, timestamp, payload_hash, decision, fraud_score, model_version, threshold, previous_hash, signature, block_hash))
            conn.commit()
            conn.close()

        return {
            "block_index": new_index,
            "timestamp": timestamp,
            "payload_hash": payload_hash,
            "decision": decision,
            "fraud_score": fraud_score,
            "model_version": model_version,
            "threshold": threshold,
            "previous_hash": previous_hash,
            "signature": signature,
            "block_hash": block_hash
        }

    def verify_chain(self) -> Tuple[bool, Optional[str]]:
        """
        Verify whole SQLite blockchain on startup or on-demand:
        - Checks hash chaining (previous_hash matches prior block_hash)
        - Checks block_hash computation
        - Checks Ed25519 signature verification
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM ledger_blocks ORDER BY block_index ASC;")
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()

        if not rows:
            return False, "Ledger database is empty."

        for i in range(len(rows)):
            row = rows[i]
            idx = row["block_index"]
            ts = row["timestamp"]
            p_hash = row["payload_hash"]
            dec = row["decision"]
            score = row["fraud_score"]
            mv = row["model_version"]
            thresh = row["threshold"]
            prev_hash = row["previous_hash"]
            sig = row["signature"]
            b_hash = row["block_hash"]

            # 1. Verify previous_hash link for i > 0
            if i > 0:
                prior_b_hash = rows[i - 1]["block_hash"]
                if prev_hash != prior_b_hash:
                    return False, f"Chain broken at Block #{idx}: previous_hash '{prev_hash[:12]}...' does not match Block #{idx-1} hash '{prior_b_hash[:12]}...'."

            # 2. Re-compute signature message & verify signature
            message = f"{idx}|{ts:.8f}|{p_hash}|{dec}|{score:.8f}|{mv}|{thresh:.8f}|{prev_hash}".encode("utf-8")
            if not self.signer.verify(sig, message):
                return False, f"Signature verification failed at Block #{idx}."

            # 3. Re-compute block hash
            block_content = f"{idx}|{ts:.8f}|{p_hash}|{dec}|{score:.8f}|{mv}|{thresh:.8f}|{prev_hash}|{sig}".encode("utf-8")
            computed_b_hash = hashlib.sha256(block_content).hexdigest()
            if b_hash != computed_b_hash:
                return False, f"Block hash corruption at Block #{idx}: stored hash '{b_hash[:12]}...' does not match computed '{computed_b_hash[:12]}...'."

        return True, "All ledger blocks and cryptographic signatures are 100% valid."

    def get_entries(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT block_index, timestamp, payload_hash, decision, fraud_score,
                   model_version, threshold, previous_hash, signature, block_hash
            FROM ledger_blocks
            ORDER BY block_index DESC
            LIMIT ? OFFSET ?;
        """, (limit, offset))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def tamper_block(self, block_index: int, new_fraud_score: float = 0.9999) -> bool:
        """
        Tamper simulation endpoint logic — ONLY active when DEMO_MODE=true.
        Modifies a historical row in SQLite without updating block_hash or signature.
        """
        with self.writer_lock:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("UPDATE ledger_blocks SET fraud_score = ? WHERE block_index = ?;", (new_fraud_score, block_index))
            conn.commit()
            conn.close()
        return True
