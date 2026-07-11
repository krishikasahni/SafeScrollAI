import os

def load_dataset(split="train"):

    dataset_path = os.path.join("dataset", split)

    video_paths = []
    labels = []

    classes = {
        "Fight": 1,
        "NonFight": 0
    }

    for class_name, label in classes.items():

        class_path = os.path.join(dataset_path, class_name)

        if not os.path.exists(class_path):
            continue

        for file in os.listdir(class_path):

            if file.lower().endswith((".avi", ".mp4", ".mov")):
                video_paths.append(os.path.join(class_path, file))
                labels.append(label)

    return video_paths, labels