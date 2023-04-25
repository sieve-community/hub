import sieve
import os
import subprocess

def get_fps(filename):
    probe_cmd = ["ffprobe", "-v", "0", "-of", "csv=p=0", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", filename]
    frame_rate = subprocess.check_output(probe_cmd).decode().strip()
    if "/" in frame_rate:
        num, denom = frame_rate.split("/")
        frame_rate = round(float(num) / float(denom), 2)
    else:
        frame_rate = float(frame_rate)

    return frame_rate

@sieve.Model(
    name="ad_segment_detector",
    python_version="3.9",
    persist_output=True,
    system_packages = [
        'autoconf',
        'libtool',
        'git',
        'build-essential',
        'libargtable2-dev',
        'libavformat-dev',
        'libsdl1.2-dev',
        'libswscale-dev',
        'ffmpeg'
    ],
    run_commands=[
        'git clone https://github.com/erikkaashoek/Comskip.git',
        'cd Comskip && ./autogen.sh && ./configure && make',
    ]
)
class ComSkip:
    def __setup__(self):
        pass

    def __predict__(self, video: sieve.Video) -> sieve.Video:
        if not os.path.exists('out'):
            os.makedirs('out')

        subprocess.run(['mv', video.path, 'out/input.mp4'])
        subprocess.run(['/Comskip/comskip', '-d', '255', 'out/input.mp4', '--output=out'])

        with open("out/input.txt", "r") as f:
            lines = f.readlines()

        fps = get_fps("out/input.mp4")

        def seconds_to_timecode(seconds):
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            seconds = seconds % 60
            return f"{hours:02}:{minutes:02}:{seconds:02}"
        # Parse the ad frame numbers
        ads = []
        for line in lines[2:]:
            start, end = map(int, line.strip().split())
            start_seconds = int(start / fps)
            end_seconds = int(end / fps)
            # Ignore ads that are too short
            if end - start < 10:
                continue

            ads.append({
                "description": "Ad segment",
                "start": seconds_to_timecode(start_seconds),
                "end": seconds_to_timecode(end_seconds),
            })

        return ads
