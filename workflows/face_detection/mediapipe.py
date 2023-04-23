import sieve
from typing import List, Dict

@sieve.workflow(name="face_detection_mediapipe")
def mediapipe_face_detection(image: sieve.Image) -> List:
    return sieve.reference("sieve/mediapipe-face-detector")(image)

@sieve.workflow(name="video_face_tracking_mediapipe")
def mediapipe_face_detection_vid(vid: sieve.Video) -> Dict:
    video_splitter = sieve.reference("sieve/video-splitter")
    frames = video_splitter(vid)
    faces = sieve.reference("sieve/mediapipe-face-detector")(frames)
    tracked = sieve.reference("sieve/sort_object_tracker")(faces)
    return tracked
