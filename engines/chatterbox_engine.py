import torch
import torchaudio
import logging
from typing import Tuple, Dict
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import numpy as np

logger = logging.getLogger(__name__)

class ChatterboxTTS:
    """Chatterbox conversational TTS engine"""
    
    def __init__(self, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.sample_rate = 16000
        self.processor = None
        self.model = None
        self.vocoder = None
        
        self.available_voices = [
            "chatterbox_female_young",
            "chatterbox_male_mature", 
            "chatterbox_child_playful",
            "chatterbox_elderly_wise",
            "chatterbox_narrator_clear"
        ]
        
        self.speaker_embeddings = {}
        self._load_model()
    
    def _load_model(self):
        """Load SpeechT5 model for conversational voices"""
        try:
            self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
            
            self._create_speaker_embeddings()
            logger.info("✅ Chatterbox SpeechT5 model loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load Chatterbox model: {e}")
            raise
    
    def _create_speaker_embeddings(self):
        """Create diverse speaker embeddings"""
        try:
            from datasets import load_dataset
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            
            # Map voices to different speakers
            speaker_mapping = {
                "chatterbox_female_young": 7306,    # Young female
                "chatterbox_male_mature": 1234,     # Mature male
                "chatterbox_child_playful": 5678,   # Child-like
                "chatterbox_elderly_wise": 9012,    # Elderly
                "chatterbox_narrator_clear": 3456   # Clear narrator
            }
            
            for voice, idx in speaker_mapping.items():
                try:
                    speaker_idx = idx % len(embeddings_dataset)
                    embedding = torch.tensor(embeddings_dataset[speaker_idx]["xvector"])
                    self.speaker_embeddings[voice] = embedding.unsqueeze(0).to(self.device)
                except:
                    # Create synthetic embedding as fallback
                    self.speaker_embeddings[voice] = torch.randn(1, 512).to(self.device)
                    
        except Exception as e:
            logger.warning(f"Failed to load embeddings: {e}, using synthetic")
            # Create all synthetic embeddings
            for voice in self.available_voices:
                self.speaker_embeddings[voice] = torch.randn(1, 512).to(self.device)
    
    def get_available_voices(self):
        return self.available_voices
    
    def synthesize(self, text: str, voice: str = "chatterbox_female_young", speed: float = 1.0) -> Tuple[torch.Tensor, int]:
        """Synthesize conversational speech"""
        
        if voice not in self.available_voices:
            voice = "chatterbox_female_young"
        
        try:
            # Prepare inputs
            inputs = self.processor(text=text, return_tensors="pt").to(self.device)
            speaker_embedding = self.speaker_embeddings[voice]
            
            # Generate speech
            with torch.no_grad():
                speech = self.model.generate_speech(
                    inputs["input_ids"], 
                    speaker_embedding, 
                    vocoder=self.vocoder
                )
            
            # Apply conversational post-processing
            speech = self._apply_conversational_effects(speech, voice)
            
            # Apply speed modification
            if speed != 1.0:
                speech = self._apply_speed_change(speech, speed)
            
            # Resample to standard rate
            if self.sample_rate != 22050:
                speech = torchaudio.functional.resample(
                    speech, self.sample_rate, 22050
                )
                output_sample_rate = 22050
            else:
                output_sample_rate = self.sample_rate
            
            return speech, output_sample_rate
            
        except Exception as e:
            logger.error(f"Chatterbox synthesis error: {e}")
            raise
    
    def _apply_conversational_effects(self, speech: torch.Tensor, voice: str) -> torch.Tensor:
        """Apply voice-specific conversational effects"""
        
        effects = {
            "chatterbox_child_playful": {"brightness": 1.2, "energy": 1.1},
            "chatterbox_elderly_wise": {"warmth": 1.3, "smoothing": 0.8},
            "chatterbox_narrator_clear": {"clarity": 1.1, "presence": 1.0},
            "chatterbox_female_young": {"brightness": 1.05, "energy": 1.0},
            "chatterbox_male_mature": {"depth": 1.1, "authority": 1.0}
        }
        
        voice_effects = effects.get(voice, {})
        
        # Apply effects (simple implementation)
        if "brightness" in voice_effects:
            speech = speech * voice_effects["brightness"]
        
        if "warmth" in voice_effects:
            speech = speech * voice_effects["warmth"]
        
        return speech
    
    def _apply_speed_change(self, audio_tensor: torch.Tensor, speed: float) -> torch.Tensor:
        """Apply speed change"""
        try:
            original_length = audio_tensor.shape[-1]
            target_length = int(original_length / speed)
            
            return torch.nn.functional.interpolate(
                audio_tensor.unsqueeze(0).unsqueeze(0),
                size=target_length,
                mode='linear',
                align_corners=False
            ).squeeze(0).squeeze(0)
        except Exception as e:
            logger.warning(f"Speed change failed: {e}")
            return audio_tensor
