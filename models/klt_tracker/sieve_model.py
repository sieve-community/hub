import sieve
from typing import Dict

@sieve.function(
    name="klt_point_tracking",
    iterator_input=True
)
def point_tracking(vids: sieve.Video, xs: int, ys: int) -> Dict:
    import cv2
    import numpy as np
    for vid, x, y in zip(vids, xs, ys):
        cap = cv2.VideoCapture(vid.path)
        point = [np.array([[x, y]], dtype=np.float32)]
        # Define LK parameters
        lk_params = dict(winSize=(21, 21),
                        maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))

        ret, old_frame = cap.read()
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

        count = 0
        while True:
            print(count)
            ret, frame = cap.read()

            if not ret:
                break

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calculate the optical flow
            new_point, status, _ = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, point[0], None, **lk_params)

            if status[0]:
                x, y = new_point.ravel()
                x, y = int(x), int(y)
                yield {
                    "x": x,
                    "y": y,
                    "frame": count,
                    "status": "Tracked point"
                }
            else:
                if not x and not y:
                    yield {
                        "status": "Unable to track point",
                        "x": -1,
                        "y": -1,
                        "frame": count
                    }
                else:
                    yield {
                        "status": "Tracked point",
                        "x": x,
                        "y": y,
                        "frame": count
                    }

            old_gray = frame_gray.copy()
            point[0] = new_point

            count += 1

        cap.release()

@sieve.function(
    name="consolidate_tracked_points",
    iterator_input=True
)
def consolidate_points(pt):
    pt = list(pt)
    sorted_by_frame = sorted(pt, key=lambda x: x["frame"])
    return sorted_by_frame

@sieve.function(
    name="point_tracking_visualize",
    python_packages=[
        "imageio==2.27.0"
    ],
    iterator_input=True,
    persist_output=True,
    system_packages=[
        "ffmpeg"
    ],
    run_commands=[
        "pip install 'imageio[ffmpeg]'"
    ],
)
def point_tracking_visualize(vids: sieve.Video, pts: list):
    import cv2
    import numpy as np
    import imageio
    # draw the points on the video and save it using imageio
    for vid, pt in zip(vids, pts):
        cap = cv2.VideoCapture(vid.path)
        ret, frame = cap.read()
        if not ret:
            break
        writer = imageio.get_writer("tracked.mp4", fps=vid.fps)
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if pt[count]["status"] == "Tracked point":
                x, y = pt[count]["x"], pt[count]["y"]
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            else:
                cv2.putText(frame, "Unable to track point", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            writer.append_data(frame)
            count += 1
        cap.release()
        writer.close()
        yield sieve.Video(path="tracked.mp4")
