import cv2 as cv
import numpy as np
import os


# ---------------- AUGMENTATION FUNCTION ----------------
def augment_sequence(base_path_seq, seq_folder_name, angles):

    original_seq_path = os.path.join(base_path_seq, seq_folder_name)

    # -------- ROTATION AUGMENTATION --------
    for angle in angles:

        aug_seq_name = seq_folder_name + f"_rot{angle}"
        aug_seq_path = os.path.join(base_path_seq, aug_seq_name)

        os.makedirs(aug_seq_path, exist_ok=True)

        for frame in sorted(os.listdir(original_seq_path)):

            if not frame.endswith(".jpg"):
                continue

            frame_path = os.path.join(original_seq_path, frame)
            img = cv.imread(frame_path)

            if img is None:
                continue

            (h, w) = img.shape[:2]
            center = (w // 2, h // 2)

            M = cv.getRotationMatrix2D(center, angle, 1.0)

            rotated = cv.warpAffine(
                img, M, (w, h),
                borderMode=cv.BORDER_REPLICATE
            )

            save_path = os.path.join(aug_seq_path, frame)
            cv.imwrite(save_path, rotated)

        print(f"Rotation augmentation {angle}° completed.")

    # -------- GRAYSCALE AUGMENTATION --------
    gray_seq_name = seq_folder_name + "_gray"
    gray_seq_path = os.path.join(base_path_seq, gray_seq_name)

    os.makedirs(gray_seq_path, exist_ok=True)

    for frame in sorted(os.listdir(original_seq_path)):

        if not frame.endswith(".jpg"):
            continue

        frame_path = os.path.join(original_seq_path, frame)
        img = cv.imread(frame_path)

        if img is None:
            continue

        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        save_path = os.path.join(gray_seq_path, frame)
        cv.imwrite(save_path, gray_img)

    print("Grayscale augmentation completed.")


# ---------------- DATASET SETUP ----------------
base_path = "dataset"
os.makedirs(base_path, exist_ok=True)

sign_name = input("Enter Sign Name: ").strip()
sign_path = os.path.join(base_path, sign_name)
os.makedirs(sign_path, exist_ok=True)

existing_base = [
    folder for folder in os.listdir(sign_path)
    if folder.startswith("base_seq_")
]

base_number = len(existing_base) + 1
base_folder_name = f"base_seq_{base_number:02d}"
base_path_seq = os.path.join(sign_path, base_folder_name)
os.makedirs(base_path_seq)

seq_folder_name = f"seq_{base_number:02d}"
seq_path = os.path.join(base_path_seq, seq_folder_name)
os.makedirs(seq_path)

print(f"Created base folder: {base_folder_name}")
print(f"Created sequence folder: {seq_folder_name}")


# ---------------- CAMERA SETUP ----------------
cap = cv.VideoCapture(0)

print("Press 'r' to record")
print("Press 'q' to quit")

recording = False
frame_count = 0
max_frames = 50  # fixed length


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_h, frame_w = frame.shape[:2]

    # -------- FIXED ROI BOX --------
    box_size = 300
    center_x = frame_w // 2
    center_y = frame_h // 2

    x1 = center_x - box_size // 2
    y1 = center_y - box_size // 2
    x2 = center_x + box_size // 2
    y2 = center_y + box_size // 2

    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    roi = frame[y1:y2, x1:x2]
    roi_resized = cv.resize(roi, (128, 128))

    cv.imshow("ROI", roi_resized)
    cv.imshow("Original", frame)

    # -------- RECORDING --------
    if recording and frame_count < max_frames:

        frame_name = f"frame_{frame_count:03d}.jpg"
        frame_path = os.path.join(seq_path, frame_name)

        cv.imwrite(frame_path, roi_resized)
        frame_count += 1

        print(f"Saved frame {frame_count}")

    if recording and frame_count == max_frames:

        print("Sequence Completed")

        # Automatic augmentation
        augment_sequence(base_path_seq, seq_folder_name, angles=[15])

        recording = False
        frame_count = 0

    key = cv.waitKey(1) & 0xFF

    if key == ord('r'):
        print("Recording started...")
        recording = True
        frame_count = 0

    if key == ord('q'):
        break


cap.release()
cv.destroyAllWindows()