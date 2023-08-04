import cv2
import mediapipe as mp
import time

# Face Mesh Detection in image
mpFaceMesh = mp.solutions.face_mesh # Face Mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces=2) # Face Mesh
mpDraw = mp.solutions.drawing_utils # Drawing Utilities

# Face Detection in image
mpFaceDetection = mp.solutions.face_detection # Face Detection
faceDetection = mpFaceDetection.FaceDetection() # Face Detection

# Drawing Specs
drawSpec = mpDraw.DrawingSpec(thickness=1, circle_radius=1)

# Video Capture
cap = cv2.VideoCapture(0)

# Give image input to Face Mesh Detection
# img = cv2.imread('images/face_mesh.jpg')

# FPS
pTime = 0
cTime = 0

while True:

    # Read image
    success, img = cap.read()

    # Convert image to RGB
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Face Detection
    results = faceDetection.process(imgRGB)

    # Face Mesh Detection
    results2 = faceMesh.process(imgRGB)

    # Draw Face Detection
    if results.detections:
        for id, detection in enumerate(results.detections):
            # mpDraw.draw_detection(img, detection)
            # print(id, detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = img.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(img, bbox, (255, 0, 255), 2)
            cv2.putText(img, f'{int(detection.score[0] * 100)}%',
                        (bbox[0], bbox[1] - 20), cv2.FONT_HERSHEY_PLAIN,
                        2, (255, 0, 255), 2)

    # Draw Face Mesh Detection
    if results2.multi_face_landmarks:
        for faceLms in results2.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS,
                                  drawSpec, drawSpec)

            # Print landmarks
            for id, lm in enumerate(faceLms.landmark):
                # print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x * iw), int(lm.y * ih)
                # print(id, x, y)

    # Calculate FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    # Display FPS
    cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (0, 255, 0), 2)

    # Display image
    cv2.imshow("Image", img)

    # Waitkey
    cv2.waitKey(1)


    