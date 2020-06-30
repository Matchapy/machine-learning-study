#%% import scipy
%matplotlib inline
from scipy.stats import dirichlet
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%%
alpha = (10,20,10)
dirichlet.pdf(x=(0.333,0.333,0.334),alpha=alpha)
dirichlet.pdf(x=(0.4,0.5,0.1),alpha=alpha)


#%%
li = []
for x in np.arange(0,1,0.01):
    for y in np.arange(0,1-x,0.01):
        z = 1-x-y
        c = dirichlet.pdf(x=(x,y,z),alpha=alpha)
        li.append([x,y,z,c])
li = np.array(li)

#%%
fig = plt.figure(figsize=(16,12))
ax = fig.add_subplot(111,projection='3d')
p = ax.scatter(li[:,0],li[:,1],li[:,2],c=li[:,3],cmap='inferno')
ax.view_init(elev=45,azim=45)
fig.colorbar(p)
plt.show()
