import sieve

@sieve.function(
    name="video-splitter",
    gpu = False,
    python_packages=[
        "ffmpeg-python==0.2.0"
    ],
    system_packages=["libgl1-mesa-glx", "libglib2.0-0", "ffmpeg"],
    python_version="3.8",
    environment_variables=[
        sieve.Env(name="FPS", default=30)
    ]
)
def VideoSplitter(video: sieve.Video) -> sieve.Image:
    import os
    fps = int(os.environ["FPS"])
    import tempfile
    temp_dir = tempfile.mkdtemp()

    import subprocess
    subprocess.call([
        'ffmpeg',
        '-i', video.path,
        '-vf', f'fps={fps}',
        f'{temp_dir}/%09d.jpg'
    ])
    import os
    filenames = os.listdir(temp_dir)
    filenames.sort()
    for i, filename in enumerate(filenames):
        print(os.path.join(temp_dir, filename), i)
        frame_number = int((video.fps / fps) * i)
        yield sieve.Image(path=os.path.join(temp_dir, filename), frame_number=frame_number, fps=fps)
