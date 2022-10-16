import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import cv2

path='./ObjectDetect/dataset/'
file=os.listdir(path)
for f in file:
    img=cv2.imread(os.path.join(path,f))
    img=cv2.resize(img,dsize=(227,227))
    #cv2.imshow('img',img)
    #cv2.waitKey(0)
    #img=img.reshape(-1,227,227,3)
    w,h=img.shape[:2]
    print(w,h)

l=[10,20,30,90,70,12,24]
print(np.argmax(l))    