import sieve
from typing import Dict

@sieve.workflow(name="u2netp_video_background_mask")
def background_mask(video: sieve.Video) -> Dict:
    images = sieve.reference('sieve/video-splitter')(video)
    masks = sieve.reference('sieve/u2netp_mask')(images)
    return sieve.reference('sieve/frame-combiner')(masks)

@sieve.workflow(name="u2netp_video_background_blur")
def background_blur(video: sieve.Video) -> Dict:
    images = sieve.reference('sieve/video-splitter')(video)
    masks = sieve.reference('sieve/u2netp_blur')(images)
    return sieve.reference('sieve/frame-combiner')(masks)
