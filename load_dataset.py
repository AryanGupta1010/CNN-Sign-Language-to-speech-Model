import os
import cv2 as cv
import numpy as np
from sklearn.model_selection import train_test_split

def load_dataset(dataset_path, sequence_length=50):
    grouped_data = {}

    gesture_names = sorted(os.listdir(dataset_path))

    for label_index, gesture in enumerate(gesture_names):
        gesture_path = os.path.join(dataset_path, gesture)
        if not os.path.isdir(gesture_path):
            continue

        for base_seq in os.listdir(gesture_path):
            base_path = os.path.join(gesture_path, base_seq)
            if not os.path.isdir(base_path):
                continue

            grouped_data[(gesture, base_seq)] = []

            for variant in os.listdir(base_path):
                variant_path = os.path.join(base_path, variant)
                if not os.path.isdir(variant_path):
                    continue

                frame_files = sorted(os.listdir(variant_path))

                count = 0
                for frame_file in frame_files:
                    if count >= sequence_length:
                        break

                    frame_path = os.path.join(variant_path, frame_file)

                    img = cv.imread(frame_path)

                    if img is None:
                        continue

                    # If grayscale (2D), convert to 3 channels
                    if len(img.shape) == 2:
                        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)

                    # If loaded as BGR (OpenCV default), convert to RGB
                    else:
                        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

                    img = img.astype("float32") / 255.0

                    grouped_data[(gesture, base_seq)].append(
                        (img, label_index)
                    )

                    count += 1

    return grouped_data

def split_grouped_data(grouped_data,test_size):
    groups=list(grouped_data.keys())

    train_groups,val_groups=train_test_split(
        groups,test_size=test_size,
        random_state=56,
        stratify=[g[0] for g in groups]
        
    )
    X_train, y_train = [], []
    X_val, y_val = [], []

    for group in train_groups:
        for frames, label in grouped_data[group]:
            X_train.append(frames)
            y_train.append(label)

    for group in val_groups:
        for frames, label in grouped_data[group]:
            X_val.append(frames)
            y_val.append(label)

    return (
        np.array(X_train), np.array(y_train),
        np.array(X_val), np.array(y_val)
    )




               
    
