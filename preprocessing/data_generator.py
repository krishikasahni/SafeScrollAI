import math
import numpy as np
from tensorflow.keras.utils import Sequence

from preprocessing.dataset_loader import load_dataset
from preprocessing.frame_extractor import extract_frames


class VideoDataGenerator(Sequence):

    def __init__(self,
                 split="train",
                 batch_size=8):

        self.video_paths, self.labels = load_dataset(split)

        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.video_paths) / self.batch_size)

    def __getitem__(self, index):

        batch_videos = self.video_paths[
            index*self.batch_size:
            (index+1)*self.batch_size
        ]

        batch_labels = self.labels[
            index*self.batch_size:
            (index+1)*self.batch_size
        ]

        X = []
        y = []

        for video, label in zip(batch_videos, batch_labels):

            frames = extract_frames(video)

            if frames is None:
                continue

            X.append(frames)
            y.append(label)

        return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)