import cv2
import time
import PoseModule as pm

cap = cv2.VideoCapture('videos/1.mp4')
pTime = 0
detector = pm.poseDetector()

# Get the width and height of the video frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a resizable window based on video dimensions
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Image", width, height)

while True:
    success, img = cap.read()
    if not success:
        break

    img = detector.findPose(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # Example of accessing specific landmark (e.g., elbow)
        elbow_x, elbow_y = lmList[14][1], lmList[14][2]
        cv2.circle(img, (elbow_x, elbow_y), 15, (0, 0, 255), cv2.FILLED)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f"FPS: {int(fps)}", (70, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

    cv2.imshow("Image", img)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
