import cv2
import dlib
import face_recognition
import numpy as np

# Load face and landmark detection models
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the images of known faces and their corresponding names
known_face_images = ["Arfan.png"]
known_face_names = ["Arfan"]

# Encode the known face images
known_face_encodings = []
for image_path in known_face_images:
    image = face_recognition.load_image_file(image_path)
    face_encoding = face_recognition.face_encodings(image)
    known_face_encodings.append(face_encoding)

# Set up the webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Detect landmarks in the face
        landmarks = landmark_predictor(gray, face)

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

        # Recognize the face
        face_encoding = face_recognition.face_encodings(frame, [face])[0]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"  # Default name for an unknown face

        # Check if there is any known face match
        if True in matches:
            index = matches.index(True)
            name = known_face_names[index]

        # Draw the face rectangle and label on the frame
        top, right, bottom, left = face.left(), face.right(), face.bottom(), face.top()
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Display the resulting frame
    cv2.imshow('Face Recognition and Eye Tracking', frame)

    # Exit the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()
