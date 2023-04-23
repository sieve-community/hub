import sieve

@sieve.workflow(name="video_lipsyncing")
def wav2lip(video: sieve.Video, audio: sieve.Audio):
    images = sieve.reference("sieve/video-splitter")(video)
    faces = sieve.reference("sieve/mediapipe-face-detector")(images)
    tracked_faces = sieve.reference("sieve/sort_object_tracker")(faces)
    return sieve.reference("sieve/wav2lip")(video, audio, tracked_faces)
