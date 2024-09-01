import cv2
import numpy as np
import time
import PoseModule as pm

# Load video file
cap = cv2.VideoCapture("video2/curls2.mp4")

# Create Pose Detector
detector = pm.poseDetector()

# Initialize variables for counting curls and direction
count = 0
dir = 0

# Variable to store previous time for FPS calculation
pTime = 0

while True:
    success, img = cap.read()

    # Check if frame was read successfully
    if not success:
        break

    # Resize the frame for processing
    img = cv2.resize(img, (1280, 720))

    # Find pose in the resized frame
    img = detector.findPose(img, False)

    # Find position of landmarks
    lmList = detector.findPosition(img, False)

    # Calculate angle and percentage
    if len(lmList) != 0:
        # Example calculation for left arm angle
        angle = detector.findAngle(img, 11, 13, 15)
        per = np.interp(angle, (210, 310), (0, 100))
        bar = np.interp(angle, (220, 310), (650, 100))

        # Count curls based on angle
        if per == 100:
            if dir == 0:
                count += 0.5
                dir = 1
        if per == 0:
            if dir == 1:
                count += 0.5
                dir = 0

    # Draw Bar
    color = (255, 0, 255) if per != 100 and per != 0 else (0, 255, 0)
    cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
    cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
    cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

    # Draw Curl Count
    cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 5)

    # Display the image with annotations
    cv2.imshow("Image", img)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
