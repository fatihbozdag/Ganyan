"""Local browser CSRF protection; remote writes require an explicit bearer token."""
import hmac
import ipaddress
import secrets
from urllib.parse import urlsplit
from flask import abort, request, session


def install_write_protection(app, settings):
    def csrf_token():
        if "csrf_token" not in session:
            session["csrf_token"] = secrets.token_urlsafe(32)
        return session["csrf_token"]

    app.jinja_env.globals["csrf_token"] = csrf_token
    app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict")

    @app.before_request
    def protect():
        if request.method not in {"POST", "PUT", "PATCH", "DELETE"}:
            return
        token = settings.mutation_token
        supplied = request.headers.get("Authorization", "")
        if token and hmac.compare_digest(supplied, "Bearer " + token):
            return  # Non-cookie API authentication is not susceptible to CSRF.
        try:
            local = ipaddress.ip_address(request.remote_addr or "").is_loopback
        except ValueError:
            local = False
        if token or not local:
            abort(401, "An authenticated mutation token is required")
        origin = request.headers.get("Origin")
        if origin:
            parsed = urlsplit(origin)
            if (parsed.scheme, parsed.netloc) != (request.scheme, request.host):
                abort(403, "Cross-origin mutation rejected")
        expected = session.get("csrf_token", "")
        supplied = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token", "")
        if not expected or not hmac.compare_digest(expected, supplied):
            abort(403, "Missing or invalid CSRF token")
