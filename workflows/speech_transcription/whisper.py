import sieve
from typing import Dict

@sieve.workflow(name="audio_transcription_whisper_2")
def audio_transcription(audio: sieve.Audio) -> Dict:
    splits = sieve.reference("sieve/audio_split_by_silence_2")(audio)
    out = sieve.reference("sieve/whisperx_2")(splits)
    return out

@sieve.workflow(name="video_transcription_whisper_2")
def video_transcription(video: sieve.Video) -> Dict:
    audio = sieve.reference("sieve/extract_audio_2")(video)
    splits = sieve.reference("sieve/audio_split_by_silence_2")(audio)
    out = sieve.reference("sieve/whisperx_2")(splits)
    return out


@sieve.workflow(name="video_transcription_whisper_2_no_split")
def video_transcription_no_split(video: sieve.Video) -> Dict:
    audio = sieve.reference("sieve/extract_audio_2")(video)
    out = sieve.reference("sieve/whisperx_2")(audio)
    return out
