import cv2
import dlib
import numpy as np

# Load face and landmark detection models
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Set up webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Detect landmarks in the face
        landmarks = predictor(gray, face)

        # Extract the coordinates of the eyes
        left_eye = []
        right_eye = []

        for n in range(36, 42):  # Left eye landmarks (36-41)
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            left_eye.append((x, y))

        for n in range(42, 48):  # Right eye landmarks (42-47)
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            right_eye.append((x, y))

        # Draw the eye contours on the frame
        cv2.polylines(frame, [np.array(left_eye)], True, (0, 255, 0), 2)
        cv2.polylines(frame, [np.array(right_eye)], True, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Eye Tracking", frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
