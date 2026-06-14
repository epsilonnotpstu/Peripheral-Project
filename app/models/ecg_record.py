import json
from app.extensions import db


class ECGRecord(db.Model):
    """
    Stores ECG data as 1-second JSON chunks (125 samples each) rather than
    individual rows — keeps SQLite row count manageable for long sessions.
    A 10-minute session creates ~600 rows instead of 75,000.
    """

    __tablename__ = "ecg_records"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("recording_sessions.id"),
        nullable=False, index=True,
    )
    timestamp = db.Column(db.Float, nullable=False)   # Unix epoch ms of chunk start
    samples = db.Column(db.Text, nullable=False)      # JSON list of float values
    bpm = db.Column(db.Float, nullable=True)
    motion_flag = db.Column(db.Boolean, default=False)

    session = db.relationship("RecordingSession", back_populates="ecg_records")

    __table_args__ = (
        db.Index("ix_ecg_records_session_ts", "session_id", "timestamp"),
    )

    def set_samples(self, samples: list) -> None:
        self.samples = json.dumps([round(float(v), 5) for v in samples])

    def get_samples(self) -> list:
        return json.loads(self.samples)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "samples": self.get_samples(),
            "bpm": self.bpm,
            "motion_flag": self.motion_flag,
        }
