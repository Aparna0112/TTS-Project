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

sys.path.append('/app')

try:
    from engines.chatterbox_engine import ChatterboxTTS
    from auth.jwt_auth import JWTAuth
except ImportError:
    from chatterbox_engine import ChatterboxTTS
    from jwt_auth import JWTAuth

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tts_engine = None
jwt_auth = JWTAuth()

def initialize_engine():
    global tts_engine
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🚀 Initializing Chatterbox on {device}")
        
        tts_engine = ChatterboxTTS(device=device)
        logger.info("✅ Chatterbox engine ready!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Engine init failed: {e}")
        return False

def audio_to_base64_mp3(audio: torch.Tensor, sample_rate: int) -> str:
    try:
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        if audio.abs().max() > 1.0:
            audio = audio / audio.abs().max() * 0.95
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), sample_rate, format="mp3")
        buffer.seek(0)
        
        mp3_bytes = buffer.getvalue()
        return base64.b64encode(mp3_bytes).decode('utf-8')
        
    except Exception as e:
        logger.error(f"Audio conversion error: {e}")
        raise

def handler(job):
    try:
        input_data = job.get("input", {})
        text = input_data.get("text", "").strip()
        jwt_token = input_data.get("jwt_token", "")
        voice = input_data.get("voice", input_data.get("voice_id", "chatterbox_female_young"))
        speed = float(input_data.get("speed", 1.0))
        
        logger.info(f"🎤 Chatterbox request: text='{text[:50]}...', voice={voice}")
        
        if not text:
            return {"success": False, "error": "Text is required"}
        
        if len(text) > 1000:
            return {"success": False, "error": "Text too long"}
        
        if jwt_token:
            payload = jwt_auth.validate_token(jwt_token)
            if not payload or not jwt_auth.has_permission(payload, "tts:synthesize"):
                return {"success": False, "error": "Invalid token or insufficient permissions"}
            user_id = payload.get("user_id", "unknown")
        else:
            user_id = "anonymous"
        
        if voice not in tts_engine.get_available_voices():
            voice = "chatterbox_female_young"
        
        if not (0.5 <= speed <= 2.0):
            return {"success": False, "error": "Speed must be between 0.5 and 2.0"}
        
        start_time = time.time()
        audio_tensor, sample_rate = tts_engine.synthesize(text, voice, speed)
        synthesis_time = time.time() - start_time
        
        audio_base64 = audio_to_base64_mp3(audio_tensor, sample_rate)
        duration = audio_tensor.shape[-1] / sample_rate
        
        result = {
            "success": True,
            "audio_data": audio_base64,
            "engine": "chatterbox",
            "voice": voice,
            "duration": duration,
            "synthesis_time": synthesis_time,
            "format": "mp3",
            "sample_rate": sample_rate,
            "user_id": user_id
        }
        
        logger.info(f"✅ Chatterbox success: {duration:.2f}s audio, {synthesis_time:.2f}s synthesis")
        return result
        
    except Exception as e:
        logger.error(f"❌ Chatterbox handler error: {e}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    logger.info("🚀 Starting Chatterbox TTS handler...")
    
    if initialize_engine():
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("❌ Failed to initialize")
        exit(1)
