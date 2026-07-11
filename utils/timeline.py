import cv2
import numpy as np

from preprocessing.frame_extractor import extract_frames


def generate_timeline(video_path, model):

    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        cap.release()
        return []

    segments = 10

    step = total_frames // segments

    timeline = []

    for i in range(segments):

        cap.set(cv2.CAP_PROP_POS_FRAMES, i * step)

        success, frame = cap.read()

        if not success:
            continue

        frames = extract_frames(video_path)

        if frames is None:
            continue

        frames = np.expand_dims(frames, axis=0)

        score = float(model.predict(frames, verbose=0)[0][0])

        if score >= 0.80:
            icon = "🔴"

        elif score >= 0.60:
            icon = "🟡"

        else:
            icon = "🟢"

        timestamp = (i * step) / fps

        timeline.append((timestamp, icon, score))

    cap.release()

    return timeline