''' файл устарел, больше не нужен. вместо него make_data
'''

import cv2
import matplotlib.pyplot as mp
import numpy as np
from common import *
import random

class RandCircClose:
    # create number of picts with 2 objects close to each other

    radius = 20 # radius of circle
    h = 200     # picture height
    w = 400     # picture width
    # create the background
    back = np.ones((h, w, 3))
    # back = np.zeros((400, 400))

    def __init__(self, number=100):
        self.num_obj = number
        self.gen_data = [None for i in range(self.num_obj)]
        for i in range (self.num_obj  ):
            back = np.ones((self.h, self.w, 3))
            x_ran = random.randint(self.radius, self.w-3*self.radius)
            y_ran = random.randint(self.radius, self.h-self.radius)
            cv2.circle(back, (x_ran, y_ran),              self.radius, red, 2)
            cv2.circle(back, (x_ran+2*self.radius, y_ran), self.radius, blue, 2)
            # cv2.imshow(f"pic {i}", back)
            # cv2.waitKey()
            self.gen_data[i] = back


if __name__ == "__main__":
    # inst = rand_circ()
    picts = RandCircClose().gen_data

    for i in range(50):
        cv2.imshow(f'pic {i}', picts[i])
        key = cv2.waitKey()
        if key == 27:
            break

    cv2.destroyAllWindows() 