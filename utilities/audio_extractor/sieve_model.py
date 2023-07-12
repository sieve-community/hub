import sieve

@sieve.function(
    name="extract_audio_2",
    python_packages=[
        "moviepy==1.0.3",
    ]
)
def extract_audio(video: sieve.Video) -> sieve.Audio:
    from moviepy.editor import VideoFileClip

    # Create the path to the output audio file
    audio_path = "audio.wav"

    # Load the video file
    video = VideoFileClip(video.path)

    # Check if the video has audio
    if not video.audio:
        raise Exception("Video has no audio")

    # Extract and save the audio
    video.audio.write_audiofile(audio_path)

    # Return the path to the audio file
    return sieve.Audio(path=audio_path)