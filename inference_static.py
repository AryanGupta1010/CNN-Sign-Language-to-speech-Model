#Inference
import cv2 as cv
import numpy as np
import tensorflow as tf
import time
from collections import deque
from soundmod import SpeechModule, process_prediction

model = tf.keras.models.load_model("gesture_model.keras")
class_names = ["Good", "Okay"]

confidence_threshold = 0.6
speech=SpeechModule()

cap = cv.VideoCapture(0)

ret, prev_frame = cap.read()
prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
prev_gray = cv.GaussianBlur(prev_gray, (5,5), 0)

prev_x, prev_y, prev_w, prev_h = 0,0,0,0
alpha = 0.7

# Frame buffer (stores last n active frames)
max_frames=10
frame_buffer = deque(maxlen=max_frames)

last_prediction_time = 0
prediction_delay = 1  # seconds

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5,5), 0)

    # Motion detection
    diff = cv.absdiff(prev_gray, gray)
    _, thresh = cv.threshold(diff, 15, 255, cv.THRESH_BINARY)

    kernel = np.ones((5,5), np.uint8)
    dilated = cv.dilate(thresh, kernel, iterations=4)

    contours, _ = cv.findContours(
        dilated, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )

    motion_detected = False

    if contours:
        largest = max(contours, key=cv.contourArea)

        if cv.contourArea(largest) > 2000:
            motion_detected = True

         #   x, y, w, h = cv.boundingRect(largest)
#
         #   # Smooth bounding box
         #   x = int(alpha*prev_x + (1-alpha)*x)
         #   y = int(alpha*prev_y + (1-alpha)*y)
         #   w = int(alpha*prev_w + (1-alpha)*w)
         #   h = int(alpha*prev_h + (1-alpha)*h)
#
         #   prev_x, prev_y, prev_w, prev_h = x,y,w,h
            x=100
            y=100
            w=200
            h=200

            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            roi = frame[y:y+h, x:x+w]
            roi_resized = cv.resize(roi,(128,128))

            # RGB preprocessing
            roi_gray = cv.cvtColor(roi_resized, cv.COLOR_BGR2GRAY)
            roi_rgb=cv.cvtColor(roi_gray,cv.COLOR_GRAY2RGB)
            normalized = roi_rgb.astype("float32") / 255.0

            # Store frame only when motion is happening
            frame_buffer.append(normalized)

    current_time = time.time()

    # Predict only if enough frames collected
    if (current_time - last_prediction_time > prediction_delay) and len(frame_buffer) == max_frames and motion_detected:

        # Take middle frame
        middle_frame = frame_buffer[int(max_frames/2)]

        input_data = np.expand_dims(middle_frame, axis=0)

        prediction = model.predict(input_data, verbose=0)
        predicted_class = np.argmax(prediction)
        confidence = prediction[0][predicted_class]

        if confidence > confidence_threshold:
            print(f"Prediction: {class_names[predicted_class]} | Confidence: {confidence:.2f}")
            process_prediction(class_names[predicted_class], confidence,speech)
        #else: print("Low confidence")

        # Clear buffer after prediction
        frame_buffer.clear()
        last_prediction_time = current_time

    prev_gray = gray.copy()

    cv.imshow("Webcam", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()