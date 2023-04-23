import sieve
from typing import Dict

@sieve.workflow(name="audio_transcription_whisper")
def audio_transcription(audio: sieve.Audio) -> Dict:
    out = sieve.reference("sieve/whisper")(audio)
    return out

@sieve.workflow(name="video_transcription_whisper")
def video_transcription(video: sieve.Video) -> Dict:
    audio = sieve.reference("sieve/extract_audio")(video)
    out = sieve.reference("sieve/whisper")(audio)
    return out
