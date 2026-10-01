import os
import sys
import time
import uuid
import pytest
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.session_utils import (
    get_remember_me_days,
    get_remember_me_lifetime_seconds,
    RoleBasedSessionInterface,
    ROLE_REMEMBER_ME_DAYS,
    REMEMBER_COOKIE_NAME,
    INACTIVITY_TIMEOUT_SECONDS,
    FARMER_REMEMBER_MAX_DAYS,
    FARMER_REMEMBER_INACTIVITY_DAYS,
    is_farmer_role
)
from app import security_service
import main
from main import app


def test_role_based_expiration_days():
    """Verify Remember Me is exclusive to Farmers (90 days) and returns 0 for non-farmer roles."""
    assert get_remember_me_days('farmer') == 90
    assert get_remember_me_lifetime_seconds('farmer') == 90 * 24 * 60 * 60
    assert is_farmer_role('farmer') is True
    assert is_farmer_role('offline_farmer') is True

    # Non-farmer roles must return 0 days (ineligible for Remember Me)
    assert get_remember_me_days('lgu') == 0
    assert get_remember_me_lifetime_seconds('lgu') == 0
    assert is_farmer_role('lgu') is False

    assert get_remember_me_days('agri_expert') == 0
    assert get_remember_me_days('agriculturist') == 0
    assert get_remember_me_lifetime_seconds('agri_expert') == 0
    assert is_farmer_role('agri_expert') is False

    assert get_remember_me_days('admin') == 0
    assert get_remember_me_days('administrator') == 0
    assert get_remember_me_lifetime_seconds('admin') == 0
    assert is_farmer_role('admin') is False

    assert get_remember_me_days('unknown_role') == 0


def test_role_based_session_interface_expiration():
    """Verify custom RoleBasedSessionInterface calculates 90 days for farmers, and None for non-farmers."""
    interface = RoleBasedSessionInterface()

    # Non-permanent session -> None
    session_mock = {'user_role': 'farmer'}
    session_mock_obj = type('MockSession', (dict,), {'permanent': False})(session_mock)
    assert interface.get_expiration_time(app, session_mock_obj) is None

    # Permanent session for farmer -> 90 days from now
    session_mock_obj.permanent = True
    now = datetime.now(timezone.utc)
    exp_farmer = interface.get_expiration_time(app, session_mock_obj)
    assert exp_farmer is not None
    assert 89 <= (exp_farmer - now).days <= 91

    # Permanent session for non-farmer (LGU / Admin) -> None (not eligible)
    session_mock_obj['user_role'] = 'lgu'
    assert interface.get_expiration_time(app, session_mock_obj) is None

    session_mock_obj['user_role'] = 'admin'
    assert interface.get_expiration_time(app, session_mock_obj) is None


def test_farmer_exclusive_token_creation():
    """Verify that only farmers can generate remember tokens, while other roles are rejected."""
    user_id = str(uuid.uuid4())
    email = "farmer_auth_test@example.com"
    
    # Farmer token creates successfully
    raw_token, expires_at = security_service.create_remember_token(user_id, email, 'farmer', 'Farmer Bob')
    assert isinstance(raw_token, str)
    assert expires_at > time.time() + (89 * 86400)

    # Non-farmer roles must raise ValueError
    with pytest.raises(ValueError):
        security_service.create_remember_token(user_id, "admin@example.com", 'admin', 'Admin User')

    with pytest.raises(ValueError):
        security_service.create_remember_token(user_id, "lgu@example.com", 'lgu', 'LGU Officer')


def test_remember_token_30_day_inactivity_expiration():
    """Verify that a farmer Remember Me token expires if inactive for more than 30 days."""
    user_id = str(uuid.uuid4())
    email = "inactivity_test@example.com"

    # 1. Create fresh token
    raw_token, expires_at = security_service.create_remember_token(user_id, email, 'farmer', 'Active Farmer')
    token_data = security_service.validate_remember_token(raw_token)
    assert token_data is not None
    assert token_data['email'] == email

    # 2. Simulate 31 days of inactivity (last_used_at set to 31 days ago)
    token_hash = security_service.hashlib.sha256(raw_token.encode('utf-8')).hexdigest()
    inactive_time = time.time() - (31 * 86400)
    with security_service._get_db() as conn:
        conn.execute("UPDATE remember_tokens SET last_used_at = ? WHERE token_hash = ?", (inactive_time, token_hash))
        conn.commit()

    # 3. Validation must reject inactive token and delete from DB
    assert security_service.validate_remember_token(raw_token) is None


def test_login_page_hides_remember_me_checkbox_and_renders_farmer_modal():
    """Verify that the login page UI hides the Remember Me checkbox from the initial form and renders the post-auth modal."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        response = client.get('/login')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        # Checkbox must NOT be in the initial login form
        assert 'name="remember_me"' not in html
        assert 'id="remember_me"' not in html
        # Post-authentication modal with 90-day and 30-day inactivity text must be in DOM
        assert 'farmer-remember-modal' in html
        assert '90 days' in html
        assert '30 days of inactivity' in html


def test_farmer_remember_me_post_auth_api():
    """Verify that the /api/auth/remember-me endpoint sets the HttpOnly cookie for farmers on opt-in."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        farmer_uid = str(uuid.uuid4())
        farmer_email = "optin_farmer@example.com"

        with client.session_transaction() as sess:
            sess['user_id'] = farmer_uid
            sess['user_email'] = farmer_email
            sess['user_role'] = 'farmer'
            sess['user_name'] = 'Opt-In Farmer'

        # Opt-in choice: 'yes'
        resp = client.post('/api/auth/remember-me', json={'choice': 'yes'})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True
        assert '/farmer/dashboard' in data['redirect_url']

        # Verify HttpOnly cookie was set
        set_cookie_headers = resp.headers.getlist('Set-Cookie')
        assert any(REMEMBER_COOKIE_NAME in h for h in set_cookie_headers)

        # Opt-out choice: 'no' -> clears cookie
        resp_no = client.post('/api/auth/remember-me', json={'choice': 'no'})
        assert resp_no.status_code == 200
        set_cookie_headers_no = resp_no.headers.getlist('Set-Cookie')
        assert any(REMEMBER_COOKIE_NAME in h and ('Max-Age=0' in h or 'Expires=' in h or '""' in h) for h in set_cookie_headers_no)


def test_hybrid_inactivity_and_farmer_remember_restoration():
    """Verify non-farmer sessions time out after 15m, while farmers are excluded from 15m inactivity."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        test_uid = str(uuid.uuid4())
        # Scenario A: Non-farmer session times out after 15m (900s)
        with client.session_transaction() as sess:
            sess['user_id'] = test_uid
            sess['user_email'] = 'timeout_agri@example.com'
            sess['user_role'] = 'agriculturist'
            sess['user_name'] = 'Agri Timeout'
            sess['last_active'] = time.time() - 950  # 950 seconds ago (> 900s)

        response = client.get('/dashboard', follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.headers.get('Location', '')
        
        # Verify non-farmer session was cleared
        with client.session_transaction() as sess:
            assert 'user_id' not in sess

        # Scenario B: Farmer session is EXCLUDED from 15m inactivity (950s does not expire session)
        farmer_uid = str(uuid.uuid4())
        with client.session_transaction() as sess:
            sess['user_id'] = farmer_uid
            sess['user_email'] = 'farmer_active@example.com'
            sess['user_role'] = 'farmer'
            sess['user_name'] = 'Farmer Active'
            sess['last_active'] = time.time() - 950  # 950 seconds ago (> 900s, but < 30 days)

        response_farmer = client.get('/dashboard', follow_redirects=False)
        assert response_farmer.status_code == 302
        assert '/farmer/dashboard' in response_farmer.headers.get('Location', '')
        with client.session_transaction() as sess:
            assert sess.get('user_id') == farmer_uid
            assert sess.get('user_email') == 'farmer_active@example.com'

        # Scenario C: Farmer session expires after 30 days of inactivity
        farmer_timeout_uid = str(uuid.uuid4())
        with client.session_transaction() as sess:
            sess['user_id'] = farmer_timeout_uid
            sess['user_email'] = 'farmer_30d_timeout@example.com'
            sess['user_role'] = 'farmer'
            sess['user_name'] = 'Farmer 30d Timeout'
            sess['last_active'] = time.time() - (31 * 86400)  # 31 days ago (> 30 days)

        response_30d = client.get('/dashboard', follow_redirects=False)
        assert response_30d.status_code == 302
        assert '/login' in response_30d.headers.get('Location', '')
        with client.session_transaction() as sess:
            assert 'user_id' not in sess


def test_logout_revokes_remember_token_and_cookie():
    """Verify logging out deletes the remember token from DB and removes cookie."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        test_uid = str(uuid.uuid4())
        raw_token, _ = security_service.create_remember_token(
            test_uid,
            'logout_farmer@example.com',
            'farmer',
            'Farmer User'
        )
        client.set_cookie(REMEMBER_COOKIE_NAME, raw_token)
        with client.session_transaction() as sess:
            sess['user_id'] = test_uid
            sess['user_email'] = 'logout_farmer@example.com'
            sess['user_role'] = 'farmer'
            sess['remember_me'] = True

        response = client.get('/logout', follow_redirects=False)
        assert response.status_code == 302
        assert '/login' in response.headers.get('Location', '')

        # Token must be revoked in DB
        assert security_service.validate_remember_token(raw_token) is None
        # Cookie must be cleared in response headers
        set_cookie_headers = response.headers.getlist('Set-Cookie')
        assert any(REMEMBER_COOKIE_NAME in h and ('Max-Age=0' in h or 'Expires=' in h or '""' in h) for h in set_cookie_headers)


def test_login_get_auto_redirects_remembered_farmer():
    """Verify GET /login auto-restores session for a farmer with a valid remember token and redirects to dashboard."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        farmer_uid = str(uuid.uuid4())
        farmer_email = "tab_close_farmer@example.com"
        raw_token, _ = security_service.create_remember_token(
            farmer_uid,
            farmer_email,
            'farmer',
            'Tab Close Farmer'
        )
        client.set_cookie(REMEMBER_COOKIE_NAME, raw_token)

        # Visit GET /login (simulating reopening browser/tab)
        response = client.get('/login', follow_redirects=False)
        assert response.status_code == 302
        assert '/farmer/dashboard' in response.headers.get('Location', '')

        # Session is restored
        with client.session_transaction() as sess:
            assert sess.get('user_id') == farmer_uid
            assert sess.get('user_email') == farmer_email
            assert sess.get('user_role') == 'farmer'
            assert sess.get('remember_me') is True
            assert sess.permanent is True

        # Role cookies are set
        set_cookie_headers = response.headers.getlist('Set-Cookie')
        assert any('cocoscan_user_role' in h for h in set_cookie_headers)


def test_login_get_auto_redirects_already_authenticated_user():
    """Verify GET /login auto-redirects an already-authenticated user to their dashboard."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        # 1. Admin user with active session
        with client.session_transaction() as sess:
            sess['user_id'] = str(uuid.uuid4())
            sess['user_email'] = 'admin@example.com'
            sess['user_role'] = 'admin'
            sess['user_name'] = 'Admin'

        response = client.get('/login', follow_redirects=False)
        assert response.status_code == 302
        assert '/admin/dashboard' in response.headers.get('Location', '')


def test_splash_recognizes_authenticated_farmer():
    """Verify GET / on splash auto-restores session for remembered farmer and sets role cookies."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        farmer_uid = str(uuid.uuid4())
        farmer_email = "splash_farmer@example.com"
        raw_token, _ = security_service.create_remember_token(
            farmer_uid,
            farmer_email,
            'farmer',
            'Splash Farmer'
        )
        client.set_cookie(REMEMBER_COOKIE_NAME, raw_token)

        response = client.get('/')
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        # Template received is_authenticated = true and dashboard_url
        assert 'isServerAuthenticated = true' in html
        assert '/farmer/dashboard' in html

        # Session is restored
        with client.session_transaction() as sess:
            assert sess.get('user_id') == farmer_uid
            assert sess.get('user_email') == farmer_email


def test_invalid_remember_token_on_login_clears_cookie():
    """Verify GET /login with an invalid remember token clears the cookie and renders login.html."""
    app.config.update(TESTING=True)
    with app.test_client() as client:
        client.set_cookie(REMEMBER_COOKIE_NAME, "completely_bogus_token_12345")

        response = client.get('/login', follow_redirects=False)
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert 'login-form' in html

        set_cookie_headers = response.headers.getlist('Set-Cookie')
        assert any(REMEMBER_COOKIE_NAME in h and ('Max-Age=0' in h or 'Expires=' in h or '""' in h) for h in set_cookie_headers)


