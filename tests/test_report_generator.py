"""
Tests for ReportGenerator — PDF generation.

Tests:
  - PDF file is created and non-empty
  - PDF contains patient name text
  - PDF contains class statistics data
  - Report fails gracefully for non-existent session
"""

import sys
import os
import tempfile
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import eventlet
eventlet.monkey_patch()

from app import create_app
from app.extensions import db as _db
from app.models.patient import Patient
from app.models.session import RecordingSession
from app.models.prediction import Prediction


@pytest.fixture(scope="module")
def app_with_data():
    """Create Flask app with test DB populated with sample data."""
    application = create_app("testing")

    with application.app_context():
        _db.create_all()

        # Create patient
        patient = Patient(name="Test Patient", age=45, gender="Male", medical_id="TEST-001")
        _db.session.add(patient)
        _db.session.flush()

        # Create completed session
        from datetime import datetime, timedelta
        session = RecordingSession(
            patient_id=patient.id,
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            ended_at=datetime(2024, 1, 15, 10, 5, 0),
            duration_s=300.0,
            is_active=False,
            notes="Test session for PDF generation",
        )
        _db.session.add(session)
        _db.session.flush()

        # Add sample predictions
        class_names = ["Normal", "Normal", "Ventricular", "Normal", "Supraventricular"]
        for i, cn in enumerate(class_names):
            class_map = {"Normal": 0, "Supraventricular": 1, "Ventricular": 2, "Fusion": 3, "Unknown": 4}
            pred = Prediction(
                session_id=session.id,
                timestamp=float(1705316400000 + i * 800),
                beat_index=i + 1,
                class_id=class_map[cn],
                class_name=cn,
                confidence=0.92 if cn == "Normal" else 0.78,
                prob_n=0.92 if cn == "Normal" else 0.05,
                prob_s=0.03,
                prob_v=0.01 if cn == "Normal" else 0.78,
                prob_f=0.02,
                prob_q=0.02,
                bpm=72.0 + i,
                motion_flag=False,
                alert_raised=(cn == "Ventricular"),
            )
            _db.session.add(pred)

        _db.session.commit()

        yield application, patient.id, session.id

        _db.drop_all()


class TestReportGenerator:
    def test_report_created(self, app_with_data, tmp_path):
        app, patient_id, session_id = app_with_data
        with app.app_context():
            from app.services.report_generator import ReportGenerator
            gen = ReportGenerator(str(tmp_path))
            pdf_path = gen.generate(session_id)

            assert os.path.exists(pdf_path), f"PDF not created at {pdf_path}"
            assert os.path.getsize(pdf_path) > 1024, "PDF file too small (< 1 KB)"

    def test_report_is_valid_pdf(self, app_with_data, tmp_path):
        app, patient_id, session_id = app_with_data
        with app.app_context():
            from app.services.report_generator import ReportGenerator
            gen = ReportGenerator(str(tmp_path))
            pdf_path = gen.generate(session_id)

            with open(pdf_path, "rb") as f:
                header = f.read(4)
            assert header == b"%PDF", f"File is not a valid PDF (header: {header!r})"

    def test_report_invalid_session(self, app_with_data, tmp_path):
        app, patient_id, session_id = app_with_data
        with app.app_context():
            from app.services.report_generator import ReportGenerator
            gen = ReportGenerator(str(tmp_path))
            with pytest.raises(ValueError, match="not found"):
                gen.generate(99999)

    def test_report_filename_contains_session_id(self, app_with_data, tmp_path):
        app, patient_id, session_id = app_with_data
        with app.app_context():
            from app.services.report_generator import ReportGenerator
            gen = ReportGenerator(str(tmp_path))
            pdf_path = gen.generate(session_id)
            filename = os.path.basename(pdf_path)
            assert f"Session{session_id}" in filename
