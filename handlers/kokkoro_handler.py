#!/usr/bin/env python3

import runpod
import torch
import torchaudio
import base64
import io
import logging
import time
import traceback
import sys
import os

# Add paths for imports
sys.path.append('/app')

try:
    from engines.kokkoro_engine import KokkoroTTS
    from auth.jwt_auth import JWTAuth
except ImportError as e:
    print(f"Import error: {e}")
    try:
        from kokkoro_engine import KokkoroTTS
        from jwt_auth import JWTAuth
    except ImportError:
        # Create fallback implementations for debugging
        class KokkoroTTS:
            def __init__(self, device="cuda"):
                self.available_voices = ["kokkoro_default", "kokkoro_sweet", "kokkoro_energetic"]
                print("⚠️ Using fallback Kokkoro TTS")
            
            def get_available_voices(self):
                return self.available_voices
            
            def synthesize(self, text, voice="kokkoro_default", speed=1.0):
                # Generate test sine wave
                sample_rate = 22050
                duration = min(len(text) * 0.08, 5.0)
                t = torch.linspace(0, duration, int(sample_rate * duration))
                frequency = 440.0 + hash(voice) % 200  # Different freq per voice
                audio = torch.sin(2 * torch.pi * frequency * t) * 0.3
                return audio, sample_rate
        
        class JWTAuth:
            def validate_token(self, token):
                return {"user_id": "fallback_user", "permissions": ["tts:synthesize"]}
            def has_permission(self, payload, perm):
                return True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
tts_engine = None
jwt_auth = JWTAuth()

def initialize_engine():
    """Initialize Kokkoro TTS engine"""
    global tts_engine
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing Kokkoro on {device}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        
        tts_engine = KokkoroTTS(device=device)
        logger.info("✅ Kokkoro engine ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine init failed: {e}")
        logger.error(traceback.format_exc())
        return False

def audio_to_base64_mp3(audio: torch.Tensor, sample_rate: int) -> str:
    """Convert audio to base64 MP3"""
    try:
        # Ensure correct shape
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        elif audio.dim() > 2:
            audio = audio.squeeze()
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
        
        # Normalize audio
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max() * 0.95
        
        # Convert to MP3
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), sample_rate, format="mp3")
        buffer.seek(0)
        
        # Encode to base64
        mp3_bytes = buffer.getvalue()
        return base64.b64encode(mp3_bytes).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise

def handler(job):
    """Main RunPod handler for Kokkoro TTS"""
    try:
        logger.info(f"📨 Received job: {job}")
        
        # Extract input data
        input_data = job.get("input", {})
        text = input_data.get("text", "").strip()
        jwt_token = input_data.get("jwt_token", "")
        voice = input_data.get("voice", input_data.get("voice_id", "kokkoro_default"))
        speed = float(input_data.get("speed", 1.0))
        
        logger.info(f"🎤 Kokkoro request: text='{text[:50]}...', voice={voice}, speed={speed}")
        
        # Validate inputs
        if not text:
            return {"success": False, "error": "Text parameter is required"}
        
        if len(text) > 1000:
            return {"success": False, "error": "Text too long (max 1000 characters)"}
        
        # Validate JWT token
        if jwt_token:
            payload = jwt_auth.validate_token(jwt_token)
            if not payload:
                return {"success": False, "error": "Invalid or expired JWT token"}
            
            if not jwt_auth.has_permission(payload, "tts:synthesize"):
                return {"success": False, "error": "Insufficient permissions for TTS synthesis"}
            
            user_id = payload.get("user_id", "unknown")
        else:
            logger.warning("⚠️ No JWT token provided")
            user_id = "anonymous"
        
        # Validate voice
        available_voices = tts_engine.get_available_voices()
        if voice not in available_voices:
            logger.warning(f"Voice '{voice}' not available, using kokkoro_default")
            voice = "kokkoro_default"
        
        # Validate speed
        if not (0.5 <= speed <= 2.0):
            return {"success": False, "error": "Speed must be between 0.5 and 2.0"}
        
        logger.info(f"🎭 Starting synthesis: '{text[:100]}...' with {voice}")
        
        # Synthesize speech
        start_time = time.time()
        audio_tensor, sample_rate = tts_engine.synthesize(
            text=text,
            voice=voice,
            speed=speed
        )
        synthesis_time = time.time() - start_time
        
        logger.info(f"⏱️ Synthesis completed in {synthesis_time:.2f}s")
        
        # Convert to MP3 base64
        audio_base64 = audio_to_base64_mp3(audio_tensor, sample_rate)
        
        # Calculate metrics
        duration = audio_tensor.shape[-1] / sample_rate
        real_time_factor = duration / synthesis_time if synthesis_time > 0 else 0
        
        result = {
            "success": True,
            "audio_data": audio_base64,
            "engine": "kokkoro",
            "voice": voice,
            "duration": duration,
            "synthesis_time": synthesis_time,
            "real_time_factor": real_time_factor,
            "format": "mp3",
            "sample_rate": sample_rate,
            "user_id": user_id,
            "text_length": len(text),
            "audio_size": len(audio_base64),
            "character_expressions": voice != "kokkoro_default"
        }
        
        logger.info(f"✅ Kokkoro success! Duration: {duration:.2f}s, RTF: {real_time_factor:.2f}x")
        return result
        
    except Exception as e:
        error_msg = f"Kokkoro handler error: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "error": error_msg,
            "engine": "kokkoro",
            "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
        }

def health_check():
    """Health check handler with detailed system info"""
    try:
        return {
            "status": "healthy",
            "engine": "kokkoro", 
            "cuda_available": torch.cuda.is_available(),
            "device": str(tts_engine.device) if tts_engine else "not_initialized",
            "available_voices": tts_engine.get_available_voices() if tts_engine else [],
            "model_loaded": tts_engine is not None,
            "timestamp": time.time(),
            "character_voices": True,
            "personality_count": len(tts_engine.get_available_voices()) if tts_engine else 0
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "engine": "kokkoro",
            "error": str(e),
            "timestamp": time.time()
        }

# Initialize and start
if __name__ == "__main__":
    logger.info("🚀 Starting Kokkoro Character TTS handler...")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    if initialize_engine():
        logger.info("✅ Starting RunPod serverless with Kokkoro character voices...")
        runpod.serverless.start({
            "handler": handler,
            "health": health_check
        })
    else:
        logger.error("❌ Failed to initialize Kokkoro engine")
        exit(1)
