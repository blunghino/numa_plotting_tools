import numpy as np 
from scipy.integrate import trapz


x = np.linspace(0,1,200)
y = np.linspace(0,1,400)
X, Y = np.meshgrid(x,y)

Z = X**2 + Y**2

int_x = trapz(Z, X, axis=1)
int_xy = trapz(int_x, Y[:,0])

sum_xy = np.sum(Z)

## should be ~ 2/3
print("integral =", int_xy)
print("sum = ", sum_xy)