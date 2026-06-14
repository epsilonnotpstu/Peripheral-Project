"""
Integration tests for REST API endpoints.
Uses Flask test client with SQLite in-memory database.

Tests:
  - GET /api/v1/health
  - POST /api/v1/patients → 201
  - GET /api/v1/patients
  - POST /api/v1/recording/start (no patient → 400, valid → 200)
  - GET /api/v1/sessions
  - DELETE /api/v1/sessions/<id>
"""

import sys
import os
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# eventlet must be monkey-patched before Flask app imports
import eventlet
eventlet.monkey_patch()

from app import create_app
from app.extensions import db as _db


@pytest.fixture(scope="module")
def app():
    application = create_app("testing")
    with application.app_context():
        _db.create_all()
        yield application
        _db.drop_all()


@pytest.fixture(scope="module")
def client(app):
    return app.test_client()


# ── Health ────────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_ok(self, client):
        resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "ok"
        assert "recording" in data
        assert "model_loaded" in data


# ── Patients ──────────────────────────────────────────────────────────────────

class TestPatients:
    def test_list_patients_empty(self, client):
        resp = client.get("/api/v1/patients")
        assert resp.status_code == 200
        assert resp.get_json() == []

    def test_create_patient_success(self, client):
        resp = client.post("/api/v1/patients", json={
            "name": "John Doe",
            "age": 55,
            "gender": "Male",
            "medical_id": "MRN-001",
        })
        assert resp.status_code == 201
        data = resp.get_json()
        assert data["name"] == "John Doe"
        assert data["age"] == 55
        assert data["id"] is not None

    def test_create_patient_no_name(self, client):
        resp = client.post("/api/v1/patients", json={"age": 30})
        assert resp.status_code == 400

    def test_list_patients_after_create(self, client):
        resp = client.get("/api/v1/patients")
        assert resp.status_code == 200
        data = resp.get_json()
        assert len(data) >= 1
        assert data[0]["name"] == "John Doe"

    def test_get_patient_by_id(self, client):
        # Get ID from list
        patients = client.get("/api/v1/patients").get_json()
        pid = patients[0]["id"]
        resp = client.get(f"/api/v1/patients/{pid}")
        assert resp.status_code == 200
        assert resp.get_json()["id"] == pid

    def test_get_patient_not_found(self, client):
        resp = client.get("/api/v1/patients/99999")
        assert resp.status_code == 404


# ── Recording ─────────────────────────────────────────────────────────────────

class TestRecording:
    def test_start_recording_no_patient_id(self, client):
        resp = client.post("/api/v1/recording/start", json={})
        assert resp.status_code == 400

    def test_start_recording_invalid_patient(self, client):
        resp = client.post("/api/v1/recording/start", json={"patient_id": 99999})
        assert resp.status_code == 404

    def test_start_recording_valid(self, client):
        patients = client.get("/api/v1/patients").get_json()
        pid = patients[0]["id"]
        resp = client.post("/api/v1/recording/start", json={"patient_id": pid})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "started"
        assert data["session_id"] is not None

    def test_stop_recording(self, client):
        resp = client.post("/api/v1/recording/stop")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "stopped"

    def test_stop_recording_when_not_running(self, client):
        resp = client.post("/api/v1/recording/stop")
        assert resp.status_code == 409


# ── Sessions ──────────────────────────────────────────────────────────────────

class TestSessions:
    def test_list_sessions(self, client):
        resp = client.get("/api/v1/sessions")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "sessions" in data
        assert "total" in data

    def test_get_session_by_id(self, client):
        sessions = client.get("/api/v1/sessions").get_json()["sessions"]
        if sessions:
            sid = sessions[0]["id"]
            resp = client.get(f"/api/v1/sessions/{sid}")
            assert resp.status_code == 200

    def test_get_session_ecg(self, client):
        sessions = client.get("/api/v1/sessions").get_json()["sessions"]
        if sessions:
            sid = sessions[0]["id"]
            resp = client.get(f"/api/v1/sessions/{sid}/ecg")
            assert resp.status_code == 200
            assert isinstance(resp.get_json(), list)

    def test_delete_completed_session(self, client):
        sessions = client.get("/api/v1/sessions").get_json()["sessions"]
        completed = [s for s in sessions if not s["is_active"]]
        if completed:
            sid = completed[0]["id"]
            resp = client.delete(f"/api/v1/sessions/{sid}")
            assert resp.status_code == 200
            # Verify deleted
            resp2 = client.get(f"/api/v1/sessions/{sid}")
            assert resp2.status_code == 404


# ── Predictions ───────────────────────────────────────────────────────────────

class TestPredictions:
    def test_get_predictions_empty(self, client):
        sessions = client.get("/api/v1/sessions").get_json()["sessions"]
        if sessions:
            sid = sessions[0]["id"]
            resp = client.get(f"/api/v1/predictions/{sid}")
            assert resp.status_code == 200
            assert isinstance(resp.get_json(), list)
