import sieve

@sieve.workflow(name = "general_object_tracking")
def wf(video: sieve.Video) -> Dict:
    images = sieve.reference('sieve/video-splitter')(video)
    yolo_outputs = sieve.reference('sieve/yolov5')(images)
    return sieve.reference('sieve/sort_object_tracker')(yolo_outputs)
