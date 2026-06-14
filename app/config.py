"""
Flask application configuration.

Settings are read from environment variables (via .env) and config.json.
CLI/environment vars take precedence over config.json defaults.
"""

import os
import json

# Project root = parent of this file's directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_config_json() -> dict:
    path = os.path.join(BASE_DIR, "config.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


_cfg = _load_config_json()


class Config:
    # ── Flask core ─────────────────────────────────────────────────────────────
    SECRET_KEY = os.environ.get("SECRET_KEY", _cfg.get("app", {}).get("secret_key", "dev-secret-key"))
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_SORT_KEYS = False

    # ── Database ───────────────────────────────────────────────────────────────
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "ecg_monitor.db"),
    )

    # ── Serial port ────────────────────────────────────────────────────────────
    SERIAL_PORT = os.environ.get("SERIAL_PORT", _cfg.get("serial", {}).get("port", "/dev/ttyUSB0"))
    SERIAL_BAUDRATE = int(os.environ.get("SERIAL_BAUDRATE", _cfg.get("serial", {}).get("baudrate", 115200)))
    SERIAL_TIMEOUT = float(_cfg.get("serial", {}).get("timeout_s", 1.0))
    SERIAL_RECONNECT_DELAY = float(_cfg.get("serial", {}).get("reconnect_delay_s", 3.0))
    SIMULATE_SERIAL = _cfg.get("serial", {}).get("simulate", False)

    # ── Signal processing ──────────────────────────────────────────────────────
    ECG_FS = int(_cfg.get("signal", {}).get("fs_hz", 125))
    ECG_BEAT_LEN = int(_cfg.get("signal", {}).get("beat_len", 187))
    BP_LOW = float(_cfg.get("signal", {}).get("bandpass_low_hz", 0.5))
    BP_HIGH = float(_cfg.get("signal", {}).get("bandpass_high_hz", 40.0))
    BP_ORDER = int(_cfg.get("signal", {}).get("bandpass_order", 4))
    NOTCH_HZ = float(_cfg.get("signal", {}).get("notch_hz", 50.0))
    NOTCH_Q = float(_cfg.get("signal", {}).get("notch_q", 30.0))
    BUFFER_SECONDS = int(_cfg.get("signal", {}).get("buffer_seconds", 5))
    RPEAK_MIN_DIST = int(_cfg.get("signal", {}).get("rpeak_min_distance_samples", 50))
    RPEAK_HEIGHT_FRAC = float(_cfg.get("signal", {}).get("rpeak_height_fraction", 0.5))

    # ── Motion detection ───────────────────────────────────────────────────────
    ACCEL_STD_THRESHOLD = float(_cfg.get("motion", {}).get("accel_std_threshold_g", 0.15))
    MOTION_WINDOW = int(_cfg.get("motion", {}).get("window_samples", 25))

    # ── Inference ──────────────────────────────────────────────────────────────
    TFLITE_MODEL_PATH = os.path.join(BASE_DIR, _cfg.get("inference", {}).get("model_path", "models/ecg_model_int8.tflite"))
    ALERT_CLASSES = _cfg.get("inference", {}).get("alert_classes", [2, 3])
    ALERT_CONFIDENCE_THRESHOLD = float(_cfg.get("inference", {}).get("alert_confidence_threshold", 0.70))
    TIMER_LOG = os.environ.get("TIMER_LOG", str(_cfg.get("inference", {}).get("timer_log", False))).lower() == "true"

    # ── Reports ────────────────────────────────────────────────────────────────
    REPORTS_DIR = os.path.join(BASE_DIR, _cfg.get("reports", {}).get("output_dir", "reports"))

    # ── UI / SocketIO ──────────────────────────────────────────────────────────
    ECG_WINDOW_SECONDS = int(_cfg.get("ui", {}).get("ecg_window_seconds", 5))
    EMIT_CHUNK_SIZE = int(_cfg.get("ui", {}).get("emit_chunk_size", 10))


class DevelopmentConfig(Config):
    DEBUG = True
    SIMULATE_SERIAL = True   # Use CSV simulator by default in dev
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "instance", "ecg_monitor_dev.db")


class ProductionConfig(Config):
    DEBUG = False
    SIMULATE_SERIAL = False
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "sqlite:///" + os.path.join(BASE_DIR, "instance", "ecg_monitor.db"),
    )


class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
    SIMULATE_SERIAL = True
    WTF_CSRF_ENABLED = False


config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
    "default": DevelopmentConfig,
}
