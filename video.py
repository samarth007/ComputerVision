import cv2
import numpy as np
cap=cv2.VideoCapture(0)

while cap.isOpened():
    ret,frame=cap.read()
    cv2.imshow('demo',frame)
    if cv2.waitKey(10) & 0xFF== ord('q'):
        break

cap.release()
cv2.destroyAllWindows()    

