from getpass import getpass

# user=int(input())
# password=getpass()
# print(user,password)

import numpy as np

w,h=0,0
data=[1,2,3,4]
box=data[0:4] * np.array([w,h,w,h])
print(box)
print(data[0:3]*np.array([2,2,1]))

print(np.linspace(start=w,stop=h,num=10))