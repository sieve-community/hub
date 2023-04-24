import sieve
from typing import Dict

@sieve.workflow(name="audio_transcription_whisper")
def audio_transcription(audio: sieve.Audio) -> Dict:
    splits = sieve.reference("sieve/audio_split_by_silence")(audio)
    out = sieve.reference("sieve/whisperx")(splits)
    return out

@sieve.workflow(name="video_transcription_whisper")
def video_transcription(video: sieve.Video) -> Dict:
    audio = sieve.reference("sieve/extract_audio")(video)
    splits = sieve.reference("sieve/audio_split_by_silence")(audio)
    out = sieve.reference("sieve/whisperx")(splits)
    return out
