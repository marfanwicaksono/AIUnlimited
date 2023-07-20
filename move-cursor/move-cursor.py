import cv2
import mediapipe as mp
import pyautogui

# Set up Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Webcam settings
width, height = 640, 480
cam = cv2.VideoCapture(1)
cam.set(3, width)
cam.set(4, height)

# Screen resolution (change this according to your screen resolution)
screen_width, screen_height = pyautogui.size()

while True:
    success, img = cam.read()
    if not success:
        break

    # Convert the image from BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the hand landmarks using Mediapipe
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                # Get the X and Y coordinates of the hand landmarks
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                # Move the cursor according to the hand gestures
                if id == 8:  # Index finger tip
                    x = int(cx * screen_width / width)
                    y = int(cy * screen_height / height)
                    pyautogui.moveTo(x, y)

    # Display the webcam feed
    cv2.imshow("Hand Tracking", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cam.release()
cv2.destroyAllWindows()
