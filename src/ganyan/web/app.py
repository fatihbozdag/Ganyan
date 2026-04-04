"""Flask application factory for the Ganyan web interface."""

from flask import Flask
from sqlalchemy.orm import sessionmaker

from ganyan.config import get_settings
from ganyan.db.session import get_session_factory


def create_app(session_factory: sessionmaker | None = None) -> Flask:
    """Create and configure the Flask application.

    Parameters
    ----------
    session_factory:
        Optional SQLAlchemy session factory.  When ``None`` (the default),
        the factory is built from :func:`ganyan.db.session.get_session_factory`.
    """
    app = Flask(__name__)

    settings = get_settings()
    app.config["SECRET_KEY"] = "dev"

    if session_factory is None:
        session_factory = get_session_factory()
    app.config["SESSION_FACTORY"] = session_factory

    from ganyan.web.routes import bp

    app.register_blueprint(bp)

    return app


def run() -> None:
    """Run the Flask development server."""
    settings = get_settings()
    app = create_app()
    app.run(host="0.0.0.0", port=settings.flask_port, debug=settings.flask_debug)
