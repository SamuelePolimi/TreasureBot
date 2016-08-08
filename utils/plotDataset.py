import matplotlib.pyplot as plt
import os
import os.path
import numpy as np

def oneDimPlot(data,label, xlabel,ylabel, ylim, std=[]):
    
    plt.plot(range(0, len(data)),data,lw=1,label=label)
        
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.title(label)
    plt.ylim(ylim[0],ylim[1])
    plt.show()



filename = raw_input("Filename of dataset: ")
label = raw_input("Title: ")
xlabel = raw_input("XLabel: ")
ylabel = raw_input("YLabel: ")

data =np.load(filename)
ylim = [np.min(data),np.max(data)]
oneDimPlot(data,label, xlabel, ylabel, ylim )
