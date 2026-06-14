"""
REST API endpoints — /api/v1/...

All endpoints return JSON. Error responses follow the format:
  {"error": "description", "status": <http_code>}
"""

import os
from datetime import datetime
from flask import request, jsonify, current_app, send_file

from app.blueprints.api import api_bp
from app.extensions import db, registry as services
from app.models.patient import Patient
from app.models.session import RecordingSession
from app.models.ecg_record import ECGRecord
from app.models.prediction import Prediction


# ── Health check ───────────────────────────────────────────────────────────────

@api_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "recording": services["serial_reader"].is_running if services["serial_reader"] else False,
        "model_loaded": services["inference_engine"] is not None,
        "serial_simulate": current_app.config.get("SIMULATE_SERIAL", False),
        "timestamp": datetime.utcnow().isoformat(),
    })


# ── Patients ───────────────────────────────────────────────────────────────────

@api_bp.route("/patients", methods=["GET"])
def list_patients():
    patients = Patient.query.order_by(Patient.created_at.desc()).all()
    return jsonify([p.to_dict() for p in patients])


@api_bp.route("/patients", methods=["POST"])
def create_patient():
    data = request.get_json(force=True)
    if not data or not data.get("name"):
        return jsonify({"error": "Patient name is required"}), 400

    patient = Patient(
        name=str(data["name"]).strip(),
        age=int(data["age"]) if data.get("age") else None,
        gender=str(data.get("gender", "")).strip() or None,
        medical_id=str(data.get("medical_id", "")).strip() or None,
        notes=str(data.get("notes", "")).strip() or None,
    )
    db.session.add(patient)
    db.session.commit()
    return jsonify(patient.to_dict()), 201


@api_bp.route("/patients/<int:patient_id>", methods=["GET"])
def get_patient(patient_id: int):
    patient = Patient.query.get_or_404(patient_id)
    d = patient.to_dict()
    d["session_count"] = patient.sessions.count()
    return jsonify(d)


# ── Recording control ──────────────────────────────────────────────────────────

@api_bp.route("/recording/start", methods=["POST"])
def start_recording():
    data = request.get_json(force=True) or {}
    patient_id = data.get("patient_id")
    if not patient_id:
        return jsonify({"error": "patient_id is required"}), 400

    patient = Patient.query.get(patient_id)
    if not patient:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404

    reader = services.get("serial_reader")
    if reader and reader.is_running:
        return jsonify({"error": "Recording already in progress"}), 409

    # Create new session
    session = RecordingSession(
        patient_id=patient_id,
        notes=str(data.get("notes", "")).strip() or None,
        is_active=True,
    )
    db.session.add(session)
    db.session.commit()

    if reader:
        reader.start(session.id)

    return jsonify({"status": "started", "session_id": session.id}), 200


@api_bp.route("/recording/stop", methods=["POST"])
def stop_recording():
    reader = services.get("serial_reader")
    if not reader or not reader.is_running:
        return jsonify({"error": "No recording in progress"}), 409

    # Find active session and close it
    active_session = RecordingSession.query.filter_by(is_active=True).first()
    if active_session:
        active_session.is_active = False
        active_session.ended_at = datetime.utcnow()
        if active_session.started_at:
            delta = active_session.ended_at - active_session.started_at
            active_session.duration_s = delta.total_seconds()
        db.session.commit()

    reader.stop()
    return jsonify({
        "status": "stopped",
        "session_id": active_session.id if active_session else None,
        "duration_s": active_session.duration_s if active_session else None,
    })


# ── Sessions ───────────────────────────────────────────────────────────────────

@api_bp.route("/sessions", methods=["GET"])
def list_sessions():
    page = int(request.args.get("page", 1))
    per_page = int(request.args.get("per_page", 20))

    sessions = (
        RecordingSession.query
        .order_by(RecordingSession.started_at.desc())
        .paginate(page=page, per_page=per_page, error_out=False)
    )
    return jsonify({
        "sessions": [s.to_dict(include_stats=True) for s in sessions.items],
        "total": sessions.total,
        "page": page,
        "pages": sessions.pages,
    })


@api_bp.route("/sessions/<int:session_id>", methods=["GET"])
def get_session(session_id: int):
    session = RecordingSession.query.get_or_404(session_id)
    return jsonify(session.to_dict(include_stats=True))


@api_bp.route("/sessions/<int:session_id>/ecg", methods=["GET"])
def get_session_ecg(session_id: int):
    """Return all ECG chunks for a session (for playback / analysis)."""
    records = (
        ECGRecord.query
        .filter_by(session_id=session_id)
        .order_by(ECGRecord.timestamp)
        .all()
    )
    return jsonify([r.to_dict() for r in records])


@api_bp.route("/sessions/<int:session_id>/report", methods=["GET"])
def generate_report(session_id: int):
    """Generate and download PDF report for a session."""
    from app.services.report_generator import ReportGenerator

    reports_dir = current_app.config.get("REPORTS_DIR", "reports")
    try:
        generator = ReportGenerator(reports_dir)
        pdf_path = generator.generate(session_id)
        return send_file(
            pdf_path,
            as_attachment=True,
            download_name=os.path.basename(pdf_path),
            mimetype="application/pdf",
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        current_app.logger.error(f"Report generation failed: {e}")
        return jsonify({"error": "Report generation failed", "detail": str(e)}), 500


@api_bp.route("/sessions/<int:session_id>", methods=["DELETE"])
def delete_session(session_id: int):
    session = RecordingSession.query.get_or_404(session_id)
    if session.is_active:
        return jsonify({"error": "Cannot delete an active recording session"}), 409
    db.session.delete(session)
    db.session.commit()
    return jsonify({"status": "deleted", "session_id": session_id})


# ── Predictions ────────────────────────────────────────────────────────────────

@api_bp.route("/predictions/<int:session_id>", methods=["GET"])
def get_predictions(session_id: int):
    limit = int(request.args.get("limit", 500))
    preds = (
        Prediction.query
        .filter_by(session_id=session_id)
        .order_by(Prediction.timestamp)
        .limit(limit)
        .all()
    )
    return jsonify([p.to_dict() for p in preds])


# ── Error handlers ─────────────────────────────────────────────────────────────

@api_bp.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found", "status": 404}), 404


@api_bp.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed", "status": 405}), 405


@api_bp.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    return jsonify({"error": "Internal server error", "status": 500}), 500
