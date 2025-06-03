import pytest
import sys
import os

# Add project root to sys.path to allow importing webapp
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Attempt to import the app, handling potential model loading issues
# by setting a flag or using a minimal app if imports fail
try:
    from webapp.app import app as flask_app
except ImportError as e:
    print(f"Conftest: Failed to import flask_app normally: {e}")
    # Fallback or re-raise, depending on desired test behavior if app is broken
    # For now, let it raise to be aware of issues.
    # If you wanted tests to run with a dummy app on import failure:
    # flask_app = Flask("dummy_app_for_broken_imports")
    # @flask_app.route('/')
    # def dummy_index(): return "Dummy App"
    raise e # Or define a minimal app for testing basic routes if main app is broken

@pytest.fixture
def app():
    # flask_app.config.update({
    # "TESTING": True,
    # })
    # You might want to set other configurations for testing, e.g., mock database
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()
