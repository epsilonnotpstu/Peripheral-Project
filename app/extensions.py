"""
Flask extension singletons — initialized here to avoid circular imports.

Import these objects into __init__.py and call .init_app(app) after app creation.
Services (serial_reader, signal_processor, etc.) are lazily initialized via
init_services() called from create_app().
"""

from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO
from flask_migrate import Migrate

db = SQLAlchemy()
socketio = SocketIO()
migrate = Migrate()

# Global service singletons — populated by init_services() in app/__init__.py
# Named 'registry' (not 'services') to avoid shadowing the app/services/ subpackage
registry: dict = {
    "serial_reader": None,
    "signal_processor": None,
    "inference_engine": None,
    "motion_detector": None,
}
