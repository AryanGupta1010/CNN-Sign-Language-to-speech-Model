1. Introduction
This project presents a real-time sign language to speech translation system developed using deep
learning techniques. The system recognizes hand gestures from a live webcam feed and converts
them into audible speech. The model is trained entirely from scratch, focusing on efficient learning
and system design under limited data conditions.
2. Data Collection and Preprocessing
The dataset was recorded using a webcam. Hand regions were isolated using contour detection
with OpenCV. The largest moving contour was assumed to correspond to the hand and extracted
from each frame. The extracted images were converted to grayscale to reduce computation while
preserving important features. Data augmentation such as rotation was applied to handle variations
in orientation. Grayscale images were expanded into three identical channels to form pseudo-RGB
images, ensuring compatibility with CNN input formats. Finally, the dataset was split into training
and validation sets.
3. Libraries Used
• OpenCV (cv2): video capture and contour detection • NumPy: numerical operations •
TensorFlow/Keras: model creation and training • gTTS: speech output These tools enabled a
lightweight and fully local implementation.
4. CNN Architecture
A custom CNN was designed with five convolutional blocks. Each block extracts increasingly
complex features. The outputs are flattened and passed through dense layers to predict gesture
labels. The architecture is kept lightweight to ensure efficient execution on CPU.
5. Training
The model was trained on the prepared dataset with validation monitoring. Augmentation helped
improve generalization given the limited data.
6. Motion-Based Inference Strategy
During inference, motion detection was performed by subtracting consecutive frames to identify
moving regions. A buffer of 50 frames was maintained, and the 25th frame was selected as a stable
representative frame. This helps reduce noise and improves prediction reliability.
7. Speech Output
Predicted gesture labels were converted into speech using gTTS, enabling real-time audio output.
8. Conclusion
The system demonstrates an effective pipeline for real-time gesture recognition using a CNN
trained from scratch. It combines efficient preprocessing, motion-based filtering, and a lightweight
model to achieve reliable performance
