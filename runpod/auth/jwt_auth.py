import jwt
import os
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

class JWTAuth:
    """Complete JWT Authentication System"""
    
    def __init__(self):
        self.secret_key = os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production")
        self.algorithm = "HS256"
        self.token_expiry_hours = int(os.getenv("JWT_EXPIRY_HOURS", "24"))
        
        if self.secret_key == "your-super-secret-jwt-key-change-in-production":
            logger.warning("⚠️ Using default JWT secret key - change in production!")
    
    def generate_token(self, user_id: str, permissions: List[str]) -> str:
        """Generate JWT token"""
        try:
            payload = {
                "user_id": user_id,
                "permissions": permissions,
                "iat": datetime.utcnow(),
                "exp": datetime.utcnow() + timedelta(hours=self.token_expiry_hours)
            }
            
            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
            logger.info(f"🔑 Generated token for user: {user_id}")
            return token
            
        except Exception as e:
            logger.error(f"Token generation failed: {e}")
            raise
    
    def validate_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Validate JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {e}")
            return None
    
    def has_permission(self, payload: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has required permission"""
        if not payload:
            return False
        
        user_permissions = payload.get("permissions", [])
        return required_permission in user_permissions
    
    def generate_test_tokens(self) -> Dict[str, str]:
        """Generate test tokens for different user types"""
        return {
            "user": self.generate_token("test_user", ["tts:synthesize"]),
            "admin": self.generate_token("admin_user", ["tts:synthesize", "admin"]),
            "api": self.generate_token("api_client", ["tts:synthesize"])
        }

# Available voices configuration
AVAILABLE_VOICES = {
    "kokkoro": [
        "kokkoro_default",
        "kokkoro_sweet", 
        "kokkoro_energetic",
        "kokkoro_calm",
        "kokkoro_shy",
        "kokkoro_determined"
    ],
    "chatterbox": [
        "chatterbox_female_young",
        "chatterbox_male_mature", 
        "chatterbox_child_playful",
        "chatterbox_elderly_wise",
        "chatterbox_narrator_clear"
    ],
    "coqui": [
        "coqui_default",
        "coqui_expressive",
        "coqui_neural",
        "coqui_multispeaker"
    ]
}

# JWT Configuration for production
PRODUCTION_JWT_CONFIG = {
    "secret_key": os.getenv("JWT_SECRET_KEY", "your-super-secret-jwt-key-change-in-production"),
    "algorithm": "HS256",
    "expiry_hours": int(os.getenv("JWT_EXPIRY_HOURS", "24")),
    "required_permissions": ["tts:synthesize"]
}

# Create global auth instance
jwt_auth = JWTAuth()
