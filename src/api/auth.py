from fastapi import APIRouter, HTTPException, Query
from kiteconnect import KiteConnect
import os
from core.state_store import StateStore

router = APIRouter(prefix="/auth/kite", tags=["Authentication"])

# Initialize Kite Connect with dummy token initially
# In production, these should be loaded securely
API_KEY = os.getenv("KITE_API_KEY")
API_SECRET = os.getenv("KITE_API_SECRET")
REDIRECT_URL = os.getenv("KITE_REDIRECT_URL", "http://localhost:8000/api/auth/kite/callback")

# Initialize StateStore for token persistence
state_store = StateStore("state/run_state.json")

def get_kite_client():
    if not API_KEY:
        raise HTTPException(status_code=500, detail="KITE_API_KEY not configured")
    return KiteConnect(api_key=API_KEY)

@router.get("/login")
def login():
    """Redirects user to Zerodha login page."""
    kite = get_kite_client()
    return {"login_url": kite.login_url()}

@router.get("/callback")
def callback(request_token: str = Query(..., description="Token from Zerodha redirect")):
    """Exchanges request_token for access_token."""
    kite = get_kite_client()
    try:
        data = kite.generate_session(request_token, api_secret=API_SECRET)
        access_token = data["access_token"]
        
        # Persist token securely (notionally)
        # In a real app, encrypt this. Here we store in our localized state store.
        state_store.set("kite_access_token", access_token)
        state_store.set("kite_public_token", data.get("public_token"))
        state_store.set("kite_user_data", {
            "user_id": data.get("user_id"),
            "user_name": data.get("user_name"),
            "login_time": data.get("login_time")
        })
        
        return {
            "status": "success", 
            "message": "Authenticated successfully", 
            "access_token": access_token # Frontend might need this for KiteTicker
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@router.get("/token")
def get_current_token():
    """Returns the current active access token (for frontend)."""
    token = state_store.get("kite_access_token")
    if not token:
        raise HTTPException(status_code=404, detail="No active session found. Please login.")
    return {"access_token": token, "api_key": API_KEY}
