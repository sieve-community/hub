import sieve
from typing import Dict

@sieve.Model(
    name="whisperx",
    gpu = True,
    python_packages=[
        "git+https://github.com/guillaumekln/faster-whisper",
        "ffmpeg-python==0.2.0",
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    run_commands=[
        "mkdir -p /root/.cache/models",
        "wget -c 'https://openaipublic.azureedge.net/main/whisper/models/81f7c96c852ee8fc832187b0132e569d6c3065a3252ed18e56effd0b6a73e524/large-v2.pt' -P /root/.cache/models"
    ]
)
class Whisper:
    def __setup__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel("large-v2", device="cuda", compute_type="float16")

    def load_audio(self, fp: str, start=None, end=None, sr: int = 16000):
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
        start_time = 0
        if hasattr(audio, "start_time") and hasattr(audio, "end_time"):
            print(f"start_time: {audio.start_time}, end_time: {audio.end_time}")
            import time
            t = time.time()
            start_time = audio.start_time
            end_time = audio.end_time
            audio_np = self.load_audio(audio.path, start=start_time, end=end_time)
            print(f"load_audio: {time.time() - t}")
            t = time.time()
            result = self.model.transcribe(audio_np, word_timestamps=True)
            print(f"transcribe: {time.time() - t}")
        else:
            t = time.time()
            result = self.model.transcribe(audio.path, word_timestamps=True)
            print(f"transcribe: {time.time() - t}")

        # print(result_aligned['word_segments'])
        out = []

        for segment in result[0]:
            words = []
            for word in segment.words:
                words.append({
                    'word': word.word,
                    'start': word.start + start_time,
                    'end': word.end + start_time,
                    'probability': word.probability
                })
            out.append({
                'text': segment.text,
                'start': segment.start + start_time,
                'end': segment.end + start_time,
                'words': words
            })

        return out
