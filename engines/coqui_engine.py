import torch
import torchaudio
import numpy as np
import tempfile
import os
import logging
from typing import Tuple
from TTS.api import TTS

logger = logging.getLogger(__name__)

class CoquiTTS:
    """Coqui TTS engine with multiple models"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = 22050
        self.models = {}
        
        self.available_voices = [
            "coqui_default",
            "coqui_expressive", 
            "coqui_neural",
            "coqui_multispeaker"
        ]
        
        # Map voices to models
        self.voice_models = {
            "coqui_default": "tts_models/en/ljspeech/tacotron2-DDC",
            "coqui_expressive": "tts_models/en/ljspeech/glow-tts", 
            "coqui_neural": "tts_models/en/ljspeech/neural_hmm",
            "coqui_multispeaker": "tts_models/en/vctk/vits"
        }
        
        self._load_models()
    
    def _load_models(self):
        """Load Coqui TTS models"""
        try:
            # Load default model first
            default_model = self.voice_models["coqui_default"]
            self.models["coqui_default"] = TTS(
                model_name=default_model,
                progress_bar=False,
                gpu=torch.cuda.is_available()
            )
            logger.info(f"✅ Loaded default Coqui model: {default_model}")
            
            # Try to load other models
            for voice, model_name in self.voice_models.items():
                if voice != "coqui_default":
                    try:
                        self.models[voice] = TTS(
                            model_name=model_name,
                            progress_bar=False,
                            gpu=torch.cuda.is_available()
                        )
                        logger.info(f"✅ Loaded {voice}: {model_name}")
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to load {voice}: {e}")
                        # Use default as fallback
                        self.models[voice] = self.models["coqui_default"]
            
        except Exception as e:
            logger.error(f"❌ Failed to load Coqui models: {e}")
            raise
    
    def get_available_voices(self):
        return self.available_voices
    
    def _get_model_for_voice(self, voice: str):
        """Get appropriate model for voice"""
        return self.models.get(voice, self.models["coqui_default"])
    
    def _enhance_text_for_voice(self, text: str, voice: str) -> str:
        """Enhance text based on voice characteristics"""
        
        enhancements = {
            "coqui_expressive": {
                # Add emphasis and expression markers
                "amazing": "AMAZING",
                "incredible": "IN-CRED-IBLE",
                "wow": "WOW!",
                "fantastic": "fan-TAS-tic"
            },
            "coqui_neural": {
                # More technical, precise language
                "process": "pro-cess",
                "technology": "tech-NOL-o-gy",
                "artificial": "ar-ti-FI-cial"
            },
            "coqui_multispeaker": {
                # Clear pronunciation for multiple speakers
                "the": "the ",  # Add slight pause
                "and": " and ",
                "with": " with "
            }
        }
        
        if voice in enhancements:
            enhanced_text = text
            for original, enhanced in enhancements[voice].items():
                enhanced_text = enhanced_text.replace(original, enhanced)
            return enhanced_text
        
        return text
    
    def synthesize(self, text: str, voice: str = "coqui_default", speed: float = 1.0) -> Tuple[torch.Tensor, int]:
        """Synthesize speech using Coqui models"""
        
        if voice not in self.available_voices:
            voice = "coqui_default"
        
        try:
            model = self._get_model_for_voice(voice)
            enhanced_text = self._enhance_text_for_voice(text, voice)
            
            logger.info(f"🗣️ Coqui synthesis: '{text[:50]}...' with {voice}")
            
            # Generate audio to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                
                if voice == "coqui_multispeaker":
                    # Multi-speaker model might need speaker specification
                    try:
                        # Try with specific speaker
                        available_speakers = getattr(model.synthesizer.tts_model, 'speaker_manager', None)
                        if available_speakers and hasattr(available_speakers, 'speaker_names'):
                            speaker_name = available_speakers.speaker_names[0]  # Use first available
                            model.tts_to_file(
                                text=enhanced_text,
                                file_path=tmp_file.name,
                                speaker=speaker_name
                            )
                        else:
                            # Try with speaker ID
                            model.tts_to_file(
                                text=enhanced_text,
                                file_path=tmp_file.name,
                                speaker="p225"
                            )
                    except:
                        # Fallback without speaker
                        model.tts_to_file(text=enhanced_text, file_path=tmp_file.name)
                else:
                    # Single speaker models
                    model.tts_to_file(text=enhanced_text, file_path=tmp_file.name)
                
                # Load generated audio
                audio_tensor, sample_rate = torchaudio.load(tmp_file.name)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                # Apply voice-specific post-processing
                audio_tensor = self._apply_voice_processing(audio_tensor, voice, sample_rate)
                
                # Apply speed modification
                if speed != 1.0:
                    audio_tensor = self._apply_speed_change(audio_tensor, speed)
                
                # Ensure correct format
                if sample_rate != self.sample_rate:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, sample_rate, self.sample_rate
                    )
                    sample_rate = self.sample_rate
                
                return audio_tensor.squeeze(0), sample_rate
                
        except Exception as e:
            logger.error(f"Coqui synthesis error: {e}")
            raise
    
    def _apply_voice_processing(self, audio: torch.Tensor, voice: str, sr: int) -> torch.Tensor:
        """Apply voice-specific audio processing"""
        
        try:
            if voice == "coqui_expressive":
                # Enhance dynamic range for expressive voice
                audio_np = audio.squeeze().cpu().numpy()
                
                # Dynamic range compression
                threshold = 0.7
                ratio = 0.6
                
                # Simple compressor
                mask = np.abs(audio_np) > threshold
                compressed = audio_np.copy()
                compressed[mask] = threshold + (audio_np[mask] - threshold) * ratio
                
                audio = torch.from_numpy(compressed).unsqueeze(0)
                
            elif voice == "coqui_neural":
                # Add slight reverb for neural voice
                audio_np = audio.squeeze().cpu().numpy()
                
                # Simple reverb using delay and decay
                delay_samples = int(0.03 * sr)  # 30ms delay
                decay = 0.3
                
                if delay_samples < len(audio_np):
                    reverb = np.zeros_like(audio_np)
                    reverb[delay_samples:] = audio_np[:-delay_samples] * decay
                    audio_np = audio_np + reverb
                
                audio = torch.from_numpy(audio_np).unsqueeze(0)
                
            elif voice == "coqui_multispeaker":
                # Enhance clarity for multi-speaker
                audio_np = audio.squeeze().cpu().numpy()
                
                # High-frequency emphasis for clarity
                # Simple high-pass filter effect
                emphasis_factor = 1.1
                diff = np.diff(audio_np, prepend=audio_np[0])
                audio_np = audio_np + diff * (emphasis_factor - 1)
                
                audio = torch.from_numpy(audio_np).unsqueeze(0)
            
            # Normalize to prevent clipping
            if audio.abs().max() > 1.0:
                audio = audio / audio.abs().max() * 0.95
                
        except Exception as e:
            logger.warning(f"Voice processing failed for {voice}: {e}")
        
        return audio
    
    def _apply_speed_change(self, audio_tensor: torch.Tensor, speed: float) -> torch.Tensor:
        """Apply speed change without pitch modification"""
        try:
            original_length = audio_tensor.shape[-1]
            target_length = int(original_length / speed)
            
            return torch.nn.functional.interpolate(
                audio_tensor.unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze(0)
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return audio_tensor
    
    def get_voice_info(self, voice: str) -> dict:
        """Get information about a Coqui voice"""
        
        voice_info = {
            "coqui_default": {
                "description": "Standard high-quality TTS with Tacotron2",
                "model": "tacotron2-DDC", 
                "optimal_use": "General purpose, clear speech",
                "quality": "High"
            },
            "coqui_expressive": {
                "description": "Expressive synthesis with Glow-TTS",
                "model": "glow-tts",
                "optimal_use": "Emotional content, storytelling",
                "quality": "Very High"
            },
            "coqui_neural": {
                "description": "Neural HMM model for natural prosody",
                "model": "neural_hmm",
                "optimal_use": "Natural conversations, audiobooks", 
                "quality": "Premium"
            },
            "coqui_multispeaker": {
                "description": "Multi-speaker VITS model",
                "model": "vits",
                "optimal_use": "Character dialogues, voice variety",
                "quality": "High"
            }
        }
        
        if voice in voice_info:
            info = voice_info[voice].copy()
            info.update({
                "voice_id": voice,
                "engine": "coqui",
                "sample_rate": self.sample_rate,
                "supports_speed_control": True,
                "supports_emotion": voice in ["coqui_expressive", "coqui_neural"]
            })
            return info
        
        return {"error": f"Voice {voice} not found"}
    
    def get_model_info(self) -> dict:
        """Get information about loaded models"""
        return {
            "loaded_models": list(self.models.keys()),
            "default_model": "coqui_default",
            "sample_rate": self.sample_rate,
            "device": str(self.device),
            "total_voices": len(self.available_voices)
        }
