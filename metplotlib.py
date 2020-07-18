import matplotlib.pyplot as plt
import numpy as np
from pylab import show, arange, sin,plot,pi

#plt.plot()
#np.array([1,2,3,4])
#plt.show()

#x=np.arange(0,5,0.1)
#y=np.sin(x)
#plt.plot(x,y)
#plt.show()

x=np.arange(0.0,2.0,0.01)
s=sin(2*pi*x)
plot(x,s)
show()
