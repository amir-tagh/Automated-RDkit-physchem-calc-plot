#!/usr/bin/env python2
#This script makes dihedral graphs for alpha,..,epsilon
 
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches  as patches
from   matplotlib import colors, ticker, cm, rc
from   matplotlib.colors import ListedColormap
from   scipy.interpolate import spline
from   mpl_toolkits.axes_grid1 import make_axes_locatable
from   mpl_toolkits.axes_grid1 import ImageGrid
import sys
import glob
import time
from matplotlib.font_manager import FontProperties
from matplotlib import rcParams
from numpy.random import randn
from numpy import genfromtxt
import seaborn as sns
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as tck
import scipy.stats as sts
from matplotlib.patches import Rectangle

#csfont = {'fontname':'Comic Sans MS'}
#hfont = {'fontname':'fantasy'}




#font0 = FontProperties()
#families = ['cursive']
#font1 = font0.copy()
#font1.set_size('large')

data_1 = open(sys.argv[1], 'r')
lines_1 = data_1.readlines()[:-1]
data_1.close()

#data_2 = open(sys.argv[2], 'r')
#lines_2 = data_2.readlines()[:-1]
#data_2.close()

x1 = []
#y1 = []



for line in lines_1:
    p = line.split()
    if float(p[0]) >= 0:
        x1.append(float(p[0]))
    #y1.append(float(p[1]))

#for line in lines_2:
#    p = line.split()
#    print('line', line)
#    y1.append(float(p[0]))
    


xv1 = np.array(x1)
#yv1 = np.array(y1)

mu, std = norm.fit(xv1)
num_bins = 10

fig, ax = plt.subplots()

# the histogram of the data
n, bins, patches = ax.hist(xv1, num_bins, edgecolor='black', linewidth=0.001,label="Drug Bank",color='darkorange',alpha=0.7)

#n, bins, patches = ax.hist(yv1, num_bins, density=True, edgecolor='black', linewidth=0.001,alpha=0.7,label="RNA focus library",color='seagreen')



#ax.set_xlabel('QED Score',fontsize=20,labelpad=9,fontname='Arial',weight='bold')
#ax.set_ylabel('Probability density',fontsize=20,labelpad=9,fontname='Arial',weight='bold')
#ax.set_title(r'Histogram of Gasteiger charges of Inforna')

#ax.set_xlim(0.0,1.0)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


#change tick thickness
ax.xaxis.set_tick_params(width=2.5)
ax.yaxis.set_tick_params(width=2.5)

#tick_spacing =0.2
#change axis thickness
ax.spines['left'].set_linewidth(4.0)
ax.spines['bottom'].set_linewidth(4.0)


#change tick thickness
ax.xaxis.set_tick_params(width=3.0)
ax.yaxis.set_tick_params(width=3.0)

ax.xaxis.set_tick_params(length=15.0)
ax.yaxis.set_tick_params(length=15.0)

plt.xticks(fontsize=24,weight='bold')
plt.yticks(fontsize=24,weight='bold')


#plt.legend()

'''
#create legend
cmap = plt.get_cmap('jet')
low = cmap(0.5)
medium =cmap(0.25)

for i in range(0,1):
    patches[i].set_facecolor(low)
for i in range(0,1):
    patches[i].set_facecolor(medium)

handles = [Rectangle((0,0),1,1,color=c,ec="k") for c in [low,medium]]
labels= ["New RNA focus library","Drug Bank"]
plt.legend(handles, labels)
'''

#ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing))
#ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))


fig.tight_layout()

fig.set_size_inches(8, 6)
plt.savefig('Fig1-main.png',dpi=300)
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
