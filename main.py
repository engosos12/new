import cv2
import numpy as np
import pyautogui

def detect_fist(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5,5),np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 2000:
        return False
    hull_points = cv2.convexHull(contour, returnPoints=False)
    if hull_points is not None and len(hull_points) > 3:
        defects = cv2.convexityDefects(contour, hull_points)
        if defects is not None:
            count_defects = 0
            for i in range(defects.shape[0]):
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                a = np.linalg.norm(np.array(end) - np.array(start))
                b = np.linalg.norm(np.array(far) - np.array(start))
                c = np.linalg.norm(np.array(end) - np.array(far))
                if b * c == 0:
                    continue
                angle = np.arccos((b*2 + c2 - a*2) / (2 * b * c))
                if angle <= np.pi / 2:
                    count_defects += 1
            if count_defects == 0:
                return True
    return False

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    left_hand = frame[100:400, 50:w//3]
    right_hand = frame[100:400, 2*w//3:w-50]
    left_fist = detect_fist(left_hand)
    right_fist = detect_fist(right_hand)
    if left_fist and right_fist:
        pyautogui.press("space")
        cv2.putText(frame, "Jump!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Running...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.rectangle(frame, (50, 100), (w//3, 400), (255, 0, 0), 2)
    cv2.rectangle(frame, (2*w//3, 100), (w-50, 400), (255, 0, 0), 2)
    cv2.putText(frame, "Left Hand", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Right Hand", (2*w//3, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow("Game Control", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()