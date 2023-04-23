import sieve

@sieve.workflow(name='talking_head_avatar_generation')
def thinplate_talking_head(driving_video: sieve.Video, driving_audio: sieve.Audio, avatar_image: sieve.Image) -> sieve.Video:
    images = sieve.reference("sieve/video-splitter")(driving_video)
    faces = sieve.reference("sieve/mediapipe-face-detector")(images)
    tracked_faces = sieve.reference("sieve/sort_object_tracker")(faces)
    synced = sieve.reference("sieve/wav2lip")(driving_video, driving_audio, tracked_faces)
    return sieve.reference("sieve/thin-plate-spline-motion-model")(avatar_image, synced)

@sieve.workflow(name='hd_talking_head_avatar_generation')
def hd_thinplate_talking_head(driving_video: sieve.Video, driving_audio: sieve.Audio, avatar_image: sieve.Image) -> sieve.Video:
    images = sieve.reference("sieve/video-splitter")(driving_video)
    faces = sieve.reference("sieve/mediapipe-face-detector")(images)
    tracked_faces = sieve.reference("sieve/sort_object_tracker")(faces)
    synced = sieve.reference("sieve/wav2lip")(driving_video, driving_audio, tracked_faces)
    return sieve.reference("sieve/hd_thin-plate-spline-motion-model")(avatar_image, synced)
