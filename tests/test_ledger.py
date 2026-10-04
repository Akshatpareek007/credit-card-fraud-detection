"""
Tests for SQLite Tamper-Evident Ledger, Deterministic Serialization, and Persistence.
"""

import os
import tempfile
import pytest
from src.ledger import AuditLedger, canonical_serialize, compute_payload_hash, Ed25519Signer


def test_canonical_serialize_determinism():
    payload1 = {"Amount": 100.5, "V1": -1.234, "Hour": 14}
    payload2 = {"Hour": 14, "V1": -1.234, "Amount": 100.5}

    s1 = canonical_serialize(payload1)
    s2 = canonical_serialize(payload2)
    assert s1 == s2
    assert "100.50000000" in s1


def test_canonical_serialize_nan_inf_rejection():
    payload_nan = {"Amount": float("nan")}
    payload_inf = {"Amount": float("inf")}

    with pytest.raises(ValueError):
        canonical_serialize(payload_nan)

    with pytest.raises(ValueError):
        canonical_serialize(payload_inf)


def test_ed25519_signer():
    signer = Ed25519Signer()
    msg = b"TEST_PAYLOAD"
    sig = signer.sign(msg)
    assert signer.verify(sig, msg) is True
    assert signer.verify(sig, b"TAMPERED_PAYLOAD") is False


def test_ledger_record_and_verify():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_ledger.db")
        ledger = AuditLedger(db_path=db_path)

        # Initial chain check (Genesis block)
        is_valid, msg = ledger.verify_chain()
        assert is_valid is True

        # Record 2 entries
        payload = {"Amount": 50.0, "V1": 0.5}
        ledger.record_entry(payload, "APPROVE", 0.05, "v2.0.0", 0.9870)
        ledger.record_entry(payload, "FRAUD", 0.99, "v2.0.0", 0.9870)

        is_valid, msg = ledger.verify_chain()
        assert is_valid is True


def test_ledger_persistence_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_persist.db")
        signer = Ed25519Signer()

        ledger1 = AuditLedger(db_path=db_path, signer=signer)
        payload = {"Amount": 120.0, "V14": -2.5}
        ledger1.record_entry(payload, "APPROVE", 0.01, "v2.0.0", 0.9870)

        # Reopen database in new instance
        ledger2 = AuditLedger(db_path=db_path, signer=signer)
        is_valid, msg = ledger2.verify_chain()
        assert is_valid is True
        entries = ledger2.get_entries()
        assert len(entries) == 2  # Genesis + 1 record


def test_ledger_tamper_detection():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_tamper.db")
        ledger = AuditLedger(db_path=db_path)

        payload = {"Amount": 200.0}
        rec = ledger.record_entry(payload, "APPROVE", 0.02, "v2.0.0", 0.9870)
        target_idx = rec["block_index"]

        # Verify clean
        assert ledger.verify_chain()[0] is True

        # Tamper historical block score directly in SQLite
        ledger.tamper_block(target_idx, new_fraud_score=0.9999)

        # Verification must FAIL
        is_valid, msg = ledger.verify_chain()
        assert is_valid is False
        assert f"Block #{target_idx}" in msg
