''' makes set of objects placed on randon distance to each other - like input data
'''
# wiki - проблема с этим файлом. выдает списсок а нужен np. массив
import cv2
import time
import matplotlib.pyplot as mp
import numpy as np
from common import *
import random

class RandCirc:
    # create number of picts with 2 objects on the distance to each other
    # class creates data via RandCircClose().np_gen_data - np array
    def __init__(   self, 
                    number=100,     # number of pictures
                    dist=0,         # dist =0 distance between objects 1- rand distance
                    radius = 20,    # circle radius
                    h= 100,         # picture height
                    w = 100         # picture width
                ):
        self.radius = radius
        self.h = h
        self.w = w
        self.num_obj = number
        self.gen_data = [None for i in range(self.num_obj)]
        for i in range (self.num_obj):
            # create the background 
            back = np.ones((self.h, self.w)) # monochrome 
            x_ran = random.randint(self.radius, self.w-4*self.radius)
            y_ran = random.randint(self.radius, self.h-self.radius)
            cv2.circle(back, (x_ran, y_ran), self.radius, red, 2)
            shift = 0
            if dist !=0: 
                shift = random.randint(10, self.w - x_ran - 3*self.radius)
            second_x = x_ran+2*self.radius + shift
            assert(second_x <= self.w-self.radius)
            cv2.circle(back, (second_x, y_ran), self.radius, green, 2)
            # cv2.imshow(f"pic {i}", back)
            # cv2.waitKey()
            self.gen_data[i] = back
                
        self.np_gen_data = np.array(self.gen_data)

if __name__ == "__main__":
    # inst = rand_circ()
    # picts = RandCirc(10000, 1).gen_data
    ts = time.time()
    picts = RandCirc(1000, 1, 20, 150, 150).gen_data
    print (f'time spend {time.time()-ts:.3}s')
    for i in range(20):
        cv2.imshow(f'pic {i}', picts[i])
        key = cv2.waitKey()
        if key == 27:
            break

    cv2.destroyAllWindows()