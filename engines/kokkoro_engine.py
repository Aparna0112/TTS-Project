import torch
import torchaudio
import numpy as np
import logging
import tempfile
import os
from typing import Tuple, Dict
from TTS.api import TTS
import librosa

logger = logging.getLogger(__name__)

class KokkoroTTS:
    """Complete Kokkoro Character TTS Engine"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050
        self.tts_model = None
        
        # Kokkoro character voices
        self.available_voices = [
            "kokkoro_default",
            "kokkoro_sweet", 
            "kokkoro_energetic",
            "kokkoro_calm",
            "kokkoro_shy",
            "kokkoro_determined"
        ]
        
        # Voice personality profiles
        self.voice_profiles = {
            "kokkoro_default": {
                "pitch_shift": 3,
                "speed_factor": 1.0,
                "brightness": 1.1,
                "emotion": "friendly"
            },
            "kokkoro_sweet": {
                "pitch_shift": 5,
                "speed_factor": 0.9,
                "brightness": 1.3,
                "emotion": "loving"
            },
            "kokkoro_energetic": {
                "pitch_shift": 4,
                "speed_factor": 1.2,
                "brightness": 1.4,
                "emotion": "excited"
            },
            "kokkoro_calm": {
                "pitch_shift": 2,
                "speed_factor": 0.8,
                "brightness": 0.9,
                "emotion": "peaceful"
            },
            "kokkoro_shy": {
                "pitch_shift": 4,
                "speed_factor": 0.85,
                "brightness": 0.8,
                "emotion": "timid"
            },
            "kokkoro_determined": {
                "pitch_shift": 1,
                "speed_factor": 1.1,
                "brightness": 1.2,
                "emotion": "confident"
            }
        }
        
        self._load_model()
    
    def _load_model(self):
        """Load TTS model"""
        try:
            # Use Tacotron2 for stable synthesis
            self.tts_model = TTS(
                model_name="tts_models/en/ljspeech/tacotron2-DDC",
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            logger.info("✅ Kokkoro TTS model loaded")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def get_available_voices(self):
        return self.available_voices
    
    def _add_character_expressions(self, text: str, voice: str) -> str:
        """Add character-specific expressions"""
        expressions = {
            "kokkoro_sweet": {
                "hello": "hello~",
                "thank you": "thank you so much~",
                "yes": "yes yes!",
                "love": "love love!"
            },
            "kokkoro_energetic": {
                "hello": "Hello there!",
                "amazing": "super amazing!",
                "great": "absolutely great!",
                "wow": "kyaa! wow!"
            },
            "kokkoro_shy": {
                "hello": "h-hello...",
                "yes": "y-yes...",
                "maybe": "m-maybe...",
                "sorry": "s-sorry..."
            },
            "kokkoro_determined": {
                "will": "will definitely",
                "can": "can absolutely",
                "fight": "will fight!",
                "protect": "will protect!"
            }
        }
        
        if voice in expressions:
            modified = text.lower()
            for old, new in expressions[voice].items():
                modified = modified.replace(old, new)
            
            # Add prefixes/suffixes based on voice
            if voice == "kokkoro_sweet":
                import random
                prefixes = ["", "Ehehe~ ", "♪ "]
                suffixes = ["~", " ♪", ""]
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                modified = f"{prefix}{modified}{suffix}"
            elif voice == "kokkoro_energetic":
                import random
                prefixes = ["", "Kyaa! ", "Wah! "]
                suffixes = ["!", "!!", ""]
                prefix = random.choice(prefixes)
                suffix = random.choice(suffixes)
                modified = f"{prefix}{modified}{suffix}"
            
            return modified.strip()
        return text
    
    def synthesize(self, text: str, voice: str = "kokkoro_default", speed: float = 1.0) -> Tuple[torch.Tensor, int]:
        """Synthesize Kokkoro voice"""
        
        if voice not in self.available_voices:
            voice = "kokkoro_default"
        
        try:
            profile = self.voice_profiles[voice]
            character_text = self._add_character_expressions(text, voice)
            
            logger.info(f"🎭 Synthesizing: '{text}' -> '{character_text}' as {voice}")
            
            # Generate base audio
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                self.tts_model.tts_to_file(
                    text=character_text,
                    file_path=tmp_file.name
                )
                
                # Load audio
                audio, sr = torchaudio.load(tmp_file.name)
                os.unlink(tmp_file.name)
            
            # Apply character voice effects
            audio = self._apply_character_effects(audio, profile, sr)
            
            # Apply speed
            total_speed = speed * profile["speed_factor"]
            if total_speed != 1.0:
                audio = self._change_speed(audio, total_speed)
            
            # Resample to target sample rate
            if sr != self.sample_rate:
                audio = torchaudio.functional.resample(audio, sr, self.sample_rate)
            
            return audio.squeeze(0), self.sample_rate
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            raise
    
    def _apply_character_effects(self, audio: torch.Tensor, profile: Dict, sr: int) -> torch.Tensor:
        """Apply character voice effects"""
        try:
            audio_np = audio.squeeze().cpu().numpy()
            
            # Pitch shifting for character voice
            if profile["pitch_shift"] != 0:
                try:
                    audio_np = librosa.effects.pitch_shift(
                        audio_np, sr=sr, n_steps=profile["pitch_shift"]
                    )
                except:
                    # Fallback: simple pitch shift using resampling
                    pitch_factor = 2 ** (profile["pitch_shift"] / 12)
                    if pitch_factor != 1.0:
                        new_length = int(len(audio_np) / pitch_factor)
                        audio_np = np.interp(
                            np.linspace(0, 1, new_length),
                            np.linspace(0, 1, len(audio_np)),
                            audio_np
                        )
            
            # Brightness adjustment
            audio_np = audio_np * profile["brightness"]
            
            # Add character-specific effects
            emotion = profile.get("emotion", "neutral")
            if emotion == "excited":
                # Add slight tremolo effect
                tremolo_rate = 6.0
                tremolo_depth = 0.1
                t = np.arange(len(audio_np)) / sr
                tremolo = 1 + tremolo_depth * np.sin(2 * np.pi * tremolo_rate * t)
                audio_np = audio_np * tremolo
            elif emotion == "timid":
                # Add breathiness (noise)
                noise_level = 0.02
                noise = np.random.normal(0, noise_level, audio_np.shape)
                audio_np = audio_np + noise
            
            # Normalize to prevent clipping
            if np.abs(audio_np).max() > 1.0:
                audio_np = audio_np / np.abs(audio_np).max() * 0.95
            
            return torch.from_numpy(audio_np).unsqueeze(0).float()
            
        except Exception as e:
            logger.warning(f"Effects failed: {e}, using original audio")
            return audio
    
    def _change_speed(self, audio: torch.Tensor, speed: float) -> torch.Tensor:
        """Change audio speed"""
        try:
            if speed == 1.0:
                return audio
            
            original_length = audio.shape[-1]
            target_length = int(original_length / speed)
            
            return torch.nn.functional.interpolate(
                audio.unsqueeze(0), 
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return audio
    
    def get_voice_info(self, voice: str) -> Dict:
        """Get information about a specific voice"""
        if voice in self.voice_profiles:
            profile = self.voice_profiles[voice]
            return {
                "voice_id": voice,
                "character": "Kokkoro",
                "emotion": profile["emotion"],
                "pitch_shift": profile["pitch_shift"],
                "speed_factor": profile["speed_factor"],
                "brightness": profile["brightness"],
                "description": self._get_voice_description(voice)
            }
        return {"error": f"Voice {voice} not found"}
    
    def _get_voice_description(self, voice: str) -> str:
        """Get personality description for voice"""
        descriptions = {
            "kokkoro_default": "Friendly and cheerful, the standard Kokkoro voice",
            "kokkoro_sweet": "Loving and affectionate, with extra sweetness and '~' expressions",
            "kokkoro_energetic": "High-energy and excited, full of enthusiasm with 'Kyaa!' exclamations", 
            "kokkoro_calm": "Gentle and soothing, peaceful demeanor for relaxation",
            "kokkoro_shy": "Soft and timid, hesitant but cute with stuttering speech",
            "kokkoro_determined": "Strong and confident, ready for action and challenges"
        }
        return descriptions.get(voice, "Unknown personality")
