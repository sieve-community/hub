import sieve
from typing import Dict

@sieve.workflow(name="scene_change_detection")
def wf(video: sieve.Video) -> Dict:
    return sieve.reference("sieve/pyscenedetect")(video)
