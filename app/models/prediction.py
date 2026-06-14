from app.extensions import db


class Prediction(db.Model):
    """Per-beat AI classification result stored for reporting and trend analysis."""

    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("recording_sessions.id"),
        nullable=False, index=True,
    )
    timestamp = db.Column(db.Float, nullable=False)      # Unix epoch ms
    beat_index = db.Column(db.Integer, nullable=True)    # Sequential beat number in session
    class_id = db.Column(db.Integer, nullable=False)     # 0–4
    class_name = db.Column(db.String(20), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    prob_n = db.Column(db.Float, nullable=True)          # P(Normal)
    prob_s = db.Column(db.Float, nullable=True)          # P(Supraventricular)
    prob_v = db.Column(db.Float, nullable=True)          # P(Ventricular)
    prob_f = db.Column(db.Float, nullable=True)          # P(Fusion)
    prob_q = db.Column(db.Float, nullable=True)          # P(Unknown)
    bpm = db.Column(db.Float, nullable=True)
    motion_flag = db.Column(db.Boolean, default=False)
    alert_raised = db.Column(db.Boolean, default=False)

    session = db.relationship("RecordingSession", back_populates="predictions")

    __table_args__ = (
        db.Index("ix_predictions_session_ts", "session_id", "timestamp"),
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "beat_index": self.beat_index,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "probabilities": {
                "N": self.prob_n,
                "S": self.prob_s,
                "V": self.prob_v,
                "F": self.prob_f,
                "Q": self.prob_q,
            },
            "bpm": self.bpm,
            "motion_flag": self.motion_flag,
            "alert_raised": self.alert_raised,
        }
