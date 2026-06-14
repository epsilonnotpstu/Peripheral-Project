"""
Development entry point for ECG Monitor.

Uses Flask-SocketIO in 'threading' async mode — no eventlet/gevent needed.
The serial reader runs as a standard threading.Thread(daemon=True).
socketio.emit() from background threads is safe in threading mode.
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()  # reads .env file if present

from app import create_app
from app.extensions import socketio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

config_name = os.environ.get("FLASK_ENV", "development")
app = create_app(config_name)

if __name__ == "__main__":
    host = app.config.get("HOST", "0.0.0.0")
    port = int(app.config.get("PORT", 5000))

    print(f"\n{'='*55}")
    print(f"  ECG Monitor — Development Server")
    print(f"  URL  : http://{host}:{port}")
    print(f"  Mode : {config_name}")
    print(f"  Sim  : {app.config.get('SIMULATE_SERIAL', True)}")
    print(f"{'='*55}\n")

    socketio.run(
        app,
        host=host,
        port=port,
        debug=app.debug,
        use_reloader=False,
        log_output=True,
        allow_unsafe_werkzeug=True,   # allows werkzeug dev server with SocketIO
    )
