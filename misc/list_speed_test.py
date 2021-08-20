# import hashlib
import cProfile
num = int(100e6)

def vect(): # 0.9 sec на 10m;   8,5s на 100m
    a = [None for i in range(num)]
    for i in range(num):
        a[i] = 1
    # print (a[:25])

def vect2(): # 2.2 sec на 10m;   24s на 100m
    a = []
    for i in range(num):
        a.append(1)

cProfile.run('vect()')
cProfile.run('vect2()')
 # аппенд работает в 3 раза медленнее
 # еще есть идея - проверить что лучше аппенд или н аппенд когда добавлять надо не единичку, 
# а какой-то большой объект. в теории большой или маленький, играть роли не должно.