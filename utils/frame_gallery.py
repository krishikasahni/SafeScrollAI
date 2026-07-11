import cv2
import numpy as np

def get_gallery_frames(video_path):

    cap = cv2.VideoCapture(video_path)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total == 0:
        cap.release()
        return []

    positions = np.linspace(0, total - 1, 6, dtype=int)

    frames = []

    for i, pos in enumerate(positions):

        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

        success, frame = cap.read()

        if success:

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000

            frames.append({

                "image": frame,

                "frame": pos,

                "time": round(timestamp, 2)

            })

    cap.release()

    return frames