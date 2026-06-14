"""
Flask application factory.

Usage:
    from app import create_app
    app = create_app("development")
"""

import os
import logging
from flask import Flask
from app.config import config
from app.extensions import db, socketio, migrate, registry


log = logging.getLogger(__name__)


def create_app(config_name: str = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_name: 'development' | 'production' | 'testing'. Falls back to
                     FLASK_ENV env var, then 'development'.
    """
    if config_name is None:
        config_name = os.environ.get("FLASK_ENV", "development")

    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "static"),
        instance_relative_config=True,
    )

    # Load configuration
    app.config.from_object(config[config_name])

    # Ensure instance and reports directories exist
    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config["REPORTS_DIR"], exist_ok=True)

    # ── Initialize extensions ─────────────────────────────────────────────────
    db.init_app(app)
    migrate.init_app(app, db)
    socketio.init_app(
        app,
        async_mode="threading",
        cors_allowed_origins="*",
        logger=False,
        engineio_logger=False,
        ping_timeout=60,
        ping_interval=25,
    )

    # ── Register blueprints ───────────────────────────────────────────────────
    from app.blueprints.main import main_bp
    from app.blueprints.api import api_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix="/api/v1")

    # Import SocketIO event handlers (registers side-effect event listeners)
    with app.app_context():
        from app.blueprints.socketio_events import events  # noqa: F401
        # Import models so SQLAlchemy knows about them
        from app.models import Patient, RecordingSession, ECGRecord, Prediction  # noqa: F401
        db.create_all()

    # ── Initialize registry ───────────────────────────────────────────────────
    with app.app_context():
        _init_registry(app)

    # ── Logging setup ──────────────────────────────────────────────────────────
    if not app.debug:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    log.info(f"ECG Monitor app created (env={config_name}, simulate={app.config['SIMULATE_SERIAL']})")
    return app


def _init_registry(app: Flask) -> None:
    """Initialize all backend services and store in the global registry dict."""
    from app.services.signal_processor import SignalProcessorService
    from app.services.motion_detector import MotionArtifactDetector

    registry["signal_processor"] = SignalProcessorService(app.config)
    registry["motion_detector"] = MotionArtifactDetector(app.config)

    # Load TFLite model — graceful fallback if model file not found
    tflite_path = app.config["TFLITE_MODEL_PATH"]
    if os.path.exists(tflite_path):
        from app.services.inference_engine import InferenceEngine
        registry["inference_engine"] = InferenceEngine(tflite_path, app.config)
        log.info(f"TFLite model loaded: {tflite_path}")
    else:
        log.warning(
            f"TFLite model not found at {tflite_path}. "
            "Run 'python ml/train.py' first to train and quantize the model. "
            "Inference will be unavailable until the model is placed there."
        )

    # SerialReader is created but not started until recording begins
    from app.services.serial_reader import SerialReaderService
    registry["serial_reader"] = SerialReaderService(app, socketio, app.config, registry)
    log.info("Services initialized.")
