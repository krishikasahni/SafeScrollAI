import cv2
import numpy as np

IMG_SIZE = 224
NUM_FRAMES = 16


def extract_frames(video_path, start_frame=None):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames < NUM_FRAMES:
        cap.release()
        return None

    # -----------------------------
    # Existing behaviour
    # -----------------------------
    if start_frame is None:

        interval = total_frames // NUM_FRAMES

        frames = []

        for i in range(NUM_FRAMES):

            cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)

            success, frame = cap.read()

            if not success:
                cap.release()
                return None

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = frame.astype(np.float32) / 255.0

            frames.append(frame)

        cap.release()

        return np.array(frames)

    # -----------------------------
    # Timeline mode
    # -----------------------------
    if start_frame + NUM_FRAMES >= total_frames:
        cap.release()
        return None

    frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for _ in range(NUM_FRAMES):

        success, frame = cap.read()

        if not success:
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
        frame = frame.astype(np.float32) / 255.0

        frames.append(frame)

    cap.release()

    return np.array(frames)