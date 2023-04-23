import sieve
import moviepy.editor as mp

@sieve.function(
    name="video-bubble",
    python_packages=[
        "git+https://github.com/Zulko/moviepy.git@master",
    ],
    iterator_input=True,
    persist_output=True
)
def screenshare(background: sieve.Video, foreground: sieve.Video) -> sieve.Video:
    import moviepy.editor as mp
    from moviepy.video.tools.drawing import circle

    for bg, fg in zip(background, foreground):
        video = mp.VideoFileClip(bg.path).without_audio()

        video_height = bg.height
        clip_height = video_height / 5

        foreground_clip = (mp.VideoFileClip(fg.path).resize(height=clip_height).margin(8).with_position(("left","bottom"))).add_mask()
        foreground_clip.mask.get_frame = lambda t: circle(screensize=(foreground_clip.w,foreground_clip.h),
                                       center=(foreground_clip.w/2,foreground_clip.h/2),
                                       radius=(foreground_clip.h - 16)/2)

        final = mp.CompositeVideoClip([video, foreground_clip])
        final.write_videofile("test.mp4")

        yield sieve.Video(path="test.mp4")
