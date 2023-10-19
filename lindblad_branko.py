import time as time
from random import seed
from random import random
import math as math
import numpy as np
import matplotlib.pyplot as plt
from sympy.physics.quantum import (Bra, Ket)
n=100; arr = [[2]*n for z in range(n)]
res = [0]*100
if __name__ == '__main__':
    b=1.0;a=0.0;ai=0.0;bi=0.0;gg=0.3;w=3.0;dp=0.0;dt=1/15;e=Ket(1);g=Ket(0);delta=1
    seed(1)
    saturation=((w**2)/4)/(delta**2+(gg**2/4)+(w**2/2))
    line= [saturation]*100
    for i in range(100):
        b=1;bi=0;a=0;ai=0
        for j in range(100):
            value = random()
            dp =2*gg*(b**2+bi**2)*dt
            if(value>dp):
                a=(a+(w/2)*bi*dt-(delta/2)*ai*dt)/math.sqrt(1-dp)
                ai=(ai-(w/2)*b*dt+(delta/2)*a*dt)/math.sqrt(1-dp)
                b=((b+(w/2)*ai*dt)-(dt*gg*b)+(delta/2)*bi*dt)/math.sqrt(1-dp)
                bi=((bi-(w/2)*a*dt)-(dt*gg*bi)-(delta/2)*b*dt)/math.sqrt(1-dp)
            else:
                ai=bi/math.sqrt(b**2+bi**2)
                a=b/math.sqrt(b**2+bi**2)
                b=0;bi=0
            print(np.abs(np.sqrt(a**2 + ai**2) + np.sqrt(b**2 + bi**2) - 1) > 0.1)
        # break
            arr[i][j]= b**2+bi**2
            arr[i][j]= b**2+bi**2
    res += np.sum(arr, axis=0)
    # for i in range(100):
    #     for j in range(100):
    #         res[i]=res[i]+arr[j][i]
    res /= 100
    # for i in range(100):
    #     res[i]=res[i]/100
    plt.plot(res,label='result')
    plt.plot(line,label='saturation')
    plt.show()