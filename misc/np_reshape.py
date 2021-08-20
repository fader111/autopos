import numpy as np
import cv2

arr = np.ones(2*100*100) 
print("Array elements: \n", arr) 
res = arr.reshape(2,100,100)

cv2.imshow('w', res[0])
cv2.waitKey()

