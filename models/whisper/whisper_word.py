import sieve
from typing import Dict

@sieve.Model(
    name="whisperx",
    gpu = True,
    python_packages=[
        "git+https://github.com/m-bain/whisperx.git",
        "ffmpeg-python==0.2.0",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "python -c 'from faster_whisper.utils import download_model; download_model(\"large-v2\")'",
    ]
)
class Whisper:
    def __setup__(self):
        import whisperx
        self.model = whisperx.load_model(
            "large-v2",
            device="cuda",
            asr_options={
                "initial_prompt": "This transcript is a single person talking. It could include filler words like 'um', 'uh', 'like', 'ah', 'like' and others words which are transcribed when they appear as a part of the transcript."
            }
        )
        self.model_a, self.metadata = whisperx.load_align_model(language_code="en", device="cuda")


    def load_audio(self, fp: str, start=None, end=None, sr: int = 32000):
        import ffmpeg
        import numpy as np

        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            if start is None and end is None:
                out, _ = (
                    ffmpeg.input(fp, threads=0)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                )
            else:
                out, _ = (
                    ffmpeg.input(fp, threads=0)
                    .filter("atrim", start=start, end=end)
                    .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                    .run(
                        cmd=["ffmpeg", "-nostdin"],
                        capture_stdout=True,
                        capture_stderr=True,
                    )
                )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0

    def __predict__(self, audio: sieve.Audio) -> Dict:
        import time
        import whisperx
        start_time = 0
        if hasattr(audio, "start_time") and hasattr(audio, "end_time"):
            print(f"start_time: {audio.start_time}, end_time: {audio.end_time}")
            import time
            t = time.time()
            start_time = audio.start_time
            end_time = audio.end_time
            audio_np = self.load_audio(audio.path, start=start_time, end=end_time)
            print(f"load_audio: {time.time() - t}")
        else:
            t = time.time()
            audio_np = whisperx.load_audio(audio.path)
            print(f"load_audio: {time.time() - t}")

        t = time.time()
        result = self.model.transcribe(audio_np, batch_size=16)
        print(f"transcribe: {time.time() - t}")
        t = time.time()
        result_aligned = whisperx.align(result["segments"], self.model_a, self.metadata, audio_np, "cuda")
        print(f"align: {time.time() - t}")
        out = []
        for segment in result_aligned["segments"]:
            words = []
            for word in segment["words"]:
                words.append({
                    'word': word["word"],
                    'start': word["start"] + start_time,
                    'end': word["end"] + start_time,
                })
            out.append({
                'start': segment["start"] + start_time,
                'end': segment["end"] + start_time,
                'text': segment["text"],
                'words': words
            })

        return out
