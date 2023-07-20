import cv2
import numpy as np

def cartoonize_image(img, gray_mode=False):
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to smooth the image while preserving edges
    gray = cv2.bilateralFilter(gray, 9, 75, 75)

    # Create an edge mask using adaptive thresholding
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 20)

    # Convert edges back to BGR color space
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Combine the smoothed image with the edge mask
    cartoon_image = cv2.bitwise_and(img, edges)

    if gray_mode:
        cartoon_image = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2GRAY)

    return cartoon_image

def webcam_cartoonize():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Flip the frame horizontally (optional, depends on your webcam orientation)
        frame = cv2.flip(frame, 1)

        # Apply the cartoonization function to the frame
        cartoon_frame = cartoonize_image(frame)

        cv2.imshow('Cartoonized', cartoon_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    webcam_cartoonize()
