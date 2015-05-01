# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def buildColorMap(target):
    """ Given an array of target values prepare a colormap 
    that can assign a proper color to each value. """
    minVal = np.min(target)
    maxVal = max(np.max(target), minVal + 0.1) # protect from the singular case
     
    # We avoid edges of the rainbow since they both look black to humans
    return lambda x: plt.cm.rainbow(0.1 + 0.8 * (x - minVal)/(maxVal - minVal))

def vis3D(data, **kwargs):
    """ Display a 3D scatter plot.

    Arguments:
        data (Bunch)        -- labelled 3D data to visualize
        
    Keyword arguments:
        dotsize (number)    -- area of the marker for each datapoint
        
    Coordinates of the dots correspond to data.data,
    colors - to data.target values. 
    """
    cm = buildColorMap(data.target)    
    
    fig = plt.figure()
    ax = Axes3D(fig)

    # Build arrays of dots of the same color    
    vals = dict()
    count = len(data.target) 
    assert count == len(data.data)
    for i in range(count):
        val = data.target[i]
        if not val in vals:
            vals[val] = [[], [], []]
        vals[val][0].append(data.data[i][0]) 
        vals[val][1].append(data.data[i][1])
        vals[val][2].append(data.data[i][2])
        
    for val in vals: 
        ax.scatter(vals[val][0], vals[val][1], vals[val][2], label = str(val),
                   marker = ".",
                   color = cm(val),
                   lw = 0, # eliminate edge
                   s = kwargs.get("dotsize", 10), # this is the area of the marker
                   depthshade = False) # eliminate confusing depth shading
  
    plt.title(kwargs.get("title", ""))  
    plt.legend()
    plt.show()

def vis2D(data, **kwargs):
    """ 2D scatter plot """
    cm = buildColorMap(data.target)    
    
    plt.figure()
    
    # Build arrays of dots of the same color    
    vals = dict()
    count = len(data.target) 
    assert count == len(data.data)
    for i in range(count):
        val = data.target[i]
        if not val in vals:
            vals[val] = [[], []]
        vals[val][0].append(data.data[i][0]) 
        vals[val][1].append(data.data[i][1])
        
    for val in vals: 
        plt.scatter(vals[val][0], vals[val][1], label = str(val),
                   marker = ".",
                   color = cm(val),
                   lw = 0, # eliminate edge
                   s = kwargs.get("dotsize", 10)) 
    
    plt.title(kwargs.get("title", ""))  
    plt.legend()
    plt.show()
   
if __name__ == "__main__":
    #TODO: unit tests
    pass