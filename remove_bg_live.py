import cv2
import numpy as np

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Define the lower and upper boundaries of the color of the object in the HSV color space
lower = np.array([0, 58, 30])
upper = np.array([33, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get a binary mask of the area of the frame within the color boundaries
    mask = cv2.inRange(hsv, lower, upper)

    # Bitwise-and the mask and the frame to get the foreground
    # fg = cv2.bitwise_and(frame, frame, mask=mask)
    # Show bg
    fg = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask))

    cv2.imshow('Frame', frame)
    cv2.imshow('Foreground', fg)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cap.release()
cv2.destroyAllWindows()