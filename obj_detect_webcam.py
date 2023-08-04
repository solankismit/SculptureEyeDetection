import cv2
import numpy as np

# Load the reference image
img1 = cv2.imread('data/dottedimg2.png',0) # reference image

# Initialize the ORB detector
orb = cv2.ORB_create()

# Find the keypoints and descriptors with ORB for the reference image
kp1, des1 = orb.detectAndCompute(img1,None)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Start the webcam feed
cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Find the keypoints and descriptors with ORB for the new image
    kp2, des2 = orb.detectAndCompute(gray,None)

    # Match descriptors
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches
    img3 = cv2.drawMatches(img1,kp1,frame,kp2,matches[:10],None, flags=2)

    # Display the resulting frame
    cv2.imshow('Matches',img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
