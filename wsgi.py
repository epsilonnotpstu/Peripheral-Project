"""
Production WSGI entry point for gunicorn + eventlet worker.

Run with:
    gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:5000 wsgi:application

Note: -w 1 is intentional — eventlet uses cooperative multitasking within a
single worker process. Multiple workers would duplicate the serial reader thread.
"""

import eventlet
eventlet.monkey_patch()

import os
from dotenv import load_dotenv
load_dotenv()

from app import create_app
from app.extensions import socketio

app = create_app(os.environ.get("FLASK_ENV", "production"))

# gunicorn looks for `application` by convention
application = socketio
