# TRY --- 1

# import cv2
# import numpy as np

# # Load the images
# img1 = cv2.imread('data/dottedimg2.png',0) # reference image
# img2 = cv2.imread('data/dottedimg3.png',0) # test image

# # Initialize the ORB detector
# orb = cv2.ORB_create(nfeatures=500)

# # Find the keypoints and descriptors with ORB
# kp1, des1 = orb.detectAndCompute(img1,None)
# kp2, des2 = orb.detectAndCompute(img2,None)

# # Create BFMatcher object
# bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# # Match descriptors
# matches = bf.match(des1,des2)

# # Sort them in the order of their distance
# matches = sorted(matches, key = lambda x:x.distance)

# # Draw first 10 matches
# n = 10
# img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:n],None, flags=2)

# cv2.imshow('Matches',img3)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




# ------------------   TRY - 2

# import cv2

# # Load the cascade
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Read the input image
# img = cv2.imread('data/dottedimg2.png')

# # Convert into grayscale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # Detect faces
# faces = face_cascade.detectMultiScale(gray, 1.1, 4)

# # Draw rectangle around the faces
# for (x, y, w, h) in faces:
#     cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

# # Display the output
# cv2.imshow('img', img)
# cv2.waitKey()



# ------------------   TRY - 3

import cv2
import numpy as np

# Load the reference image
img1 = cv2.imread('data/dottedimg2.png',0) # reference image

# Initialize the FAST detector
fast = cv2.FastFeatureDetector_create()

# Initialize the BRIEF descriptor
brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()

# Detect keypoints with FAST
kp1 = fast.detect(img1,None)

# Compute descriptors with BRIEF
kp1, des1 = brief.compute(img1, kp1)

# Create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

# Start the webcam feed
# cap = cv2.VideoCapture(0)
while(True):
    # Capture frame-by-frame
    # ret, frame = cap.read()
    frame = cv2.imread('data/dottedimg3.png')
    # if not ret:
    #     break

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints with FAST
    kp2 = fast.detect(gray, None)

    # Compute descriptors with BRIEF
    kp2, des2 = brief.compute(gray, kp2)

    # Match descriptors.
    matches = bf.match(des1,des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    # Draw first 10 matches.
    img3 = cv2.drawMatches(img1,kp1,gray,kp2,matches[:10], None, flags=2)

    # Display the resulting frame
    cv2.imshow('Matches',img3)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
# cap.release()
cv2.destroyAllWindows()
