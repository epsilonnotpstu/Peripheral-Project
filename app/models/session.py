from datetime import datetime
from app.extensions import db


class RecordingSession(db.Model):
    __tablename__ = "recording_sessions"

    id = db.Column(db.Integer, primary_key=True)
    patient_id = db.Column(db.Integer, db.ForeignKey("patients.id"), nullable=False)
    started_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    ended_at = db.Column(db.DateTime, nullable=True)
    duration_s = db.Column(db.Float, nullable=True)
    notes = db.Column(db.Text, nullable=True)
    is_active = db.Column(db.Boolean, default=True, nullable=False)

    patient = db.relationship("Patient", back_populates="sessions")
    ecg_records = db.relationship(
        "ECGRecord", back_populates="session",
        lazy="dynamic", cascade="all, delete-orphan",
    )
    predictions = db.relationship(
        "Prediction", back_populates="session",
        lazy="dynamic", cascade="all, delete-orphan",
    )

    def to_dict(self, include_stats: bool = False) -> dict:
        d = {
            "id": self.id,
            "patient_id": self.patient_id,
            "patient_name": self.patient.name if self.patient else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_s": self.duration_s,
            "notes": self.notes,
            "is_active": self.is_active,
        }
        if include_stats:
            preds = list(self.predictions.all())
            d["total_beats"] = len(preds)
            d["dominant_class"] = self._dominant_class(preds)
            d["avg_bpm"] = round(
                sum(p.bpm for p in preds if p.bpm) / max(1, sum(1 for p in preds if p.bpm)),
                1,
            )
        return d

    @staticmethod
    def _dominant_class(preds: list) -> str:
        if not preds:
            return "N/A"
        counts = {}
        for p in preds:
            counts[p.class_name] = counts.get(p.class_name, 0) + 1
        return max(counts, key=counts.get)
