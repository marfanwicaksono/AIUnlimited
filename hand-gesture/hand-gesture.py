import cv2
import mediapipe as mp

def detect_hand_gestures():
    mp_hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(1)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB and process it with Mediapipe
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = mp_hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for point in mp.solutions.hands.HandLandmark:
                    x, y = int(hand_landmarks.landmark[point].x * frame.shape[1]), int(hand_landmarks.landmark[point].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)

        cv2.imshow('Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_hand_gestures()
