import sieve
from typing import Dict

@sieve.workflow(
    name="ad_segment_detection"
)
def ad_segment_detection(video: sieve.Video) -> Dict:
    return sieve.reference("ad_segment_detector")(video)
