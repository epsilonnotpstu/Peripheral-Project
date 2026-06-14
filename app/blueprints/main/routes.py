from flask import render_template
from app.blueprints.main import main_bp
from app.models.patient import Patient
from app.models.session import RecordingSession
from app.extensions import registry as services


@main_bp.route("/", methods=["GET"])
def index():
    patients = Patient.query.order_by(Patient.name).all()
    recent_sessions = (
        RecordingSession.query
        .order_by(RecordingSession.started_at.desc())
        .limit(20)
        .all()
    )
    return render_template(
        "dashboard.html",
        patients=patients,
        sessions=recent_sessions,
        model_loaded=services.get("inference_engine") is not None,
        simulate_mode=True,   # passed to JS for UI hints
    )
