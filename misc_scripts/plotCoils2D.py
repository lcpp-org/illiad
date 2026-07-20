 ############
 ## IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from illiad.utilities.coordtrans import XYZ_to_RTP
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

hi_res = [ 191, 180, 360 ] # dr=0.001m., dtheta=2deg., dphi=1deg.
mesh_size = hi_res
from scipy.signal import argrelextrema

def main():
    Rmajor = 0.72 #[m]
    Rminor = 0.19 #[m]

    ## READ COIL INPUT FILE
    coilfile = "input_files/coils.wega_with_VFCoils"
    coildata = pd.read_csv(
    coilfile,
    header=None,
    skiprows=3,
    index_col=None,
    delim_whitespace=True,
    names=range(6)) #irregularly-sized rows, pad with NaN's

    ## PARSE COIL DATA
    coil_delim = coildata.loc[coildata[5].isnull()==False].index.values.tolist() # find index of rows with coil tag

    coiltype = coildata.iloc[coil_delim][5].values
    turns = [coildata.iloc[i-1][3] for i in coil_delim]

    mycoils = [None]*len(coil_delim)
    for i, dum in enumerate(coil_delim):
        if i==0:
            mycoils[i] = coildata.iloc[:coil_delim[i], 0:4]
        else:
            mycoils[i] = coildata.iloc[coil_delim[i-1]+1:coil_delim[i], 0:4]

    fig = plt.figure()#figsize=(12.8, 9.6)) #default: 6.4 x 4.8
    ax = fig.add_subplot()#projection='3d')
    
    for n, coil in enumerate(mycoils):

        coilpts = np.asarray(coil, dtype=np.float64)
        filament = np.ascontiguousarray(coilpts.T[:3])
        
        r = np.zeros(filament.shape[1])
        t = np.zeros(filament.shape[1])
        p = np.zeros(filament.shape[1])

        for i,point in enumerate(filament.T): #filament has all of the points in XYZ coordinate system
            radius, theta, phi = XYZ_to_RTP(np.ascontiguousarray(point), Rmajor)

            r[i] = radius
            t[i] = np.degrees(theta)
            if t[i]>=180: #Adjusts the theta so the range is (-180,180)
                t[i] -=360
            p[i] = np.degrees(-phi) +360# Negative to flip the orientation to clockwise to match the helical coils around HIDRA
        
        if coiltype[n] == 'Helix':
            if float(coilpts[0][3]) < 0:    c = 'r'; firstTime=0 #Negative current is in red
            elif float(coilpts[0][3]) > 0:  c = 'g'; firstTime=0 #Positive current is in green
            
            #This code is used to cut up the helical coils so there aren't huge diagonal lines on the graph. It does this by finding the edges\
            # where the coil cross a plane, splits it there and then graphs each individual split part for the filament\ 
            # Edges is a list of these points where it crosses the 0-360 plane; the dots are to clarify which way the coils are going (start -> stop)

            highends = np.append(argrelextrema(p, np.greater), argrelextrema(p, np.less))
            
            lowends = np.append(argrelextrema(t, np.greater), argrelextrema(t, np.less))
            
            edges = np.append(highends, lowends)
            edges = np.sort(edges)
            edges = np.append(0, np.append(edges, len(p)-1))
            edges = np.array([int(edge) for edge in edges])
            
            
            for i in range(len(edges)-1):
                start = edges[i]
                stop = edges[i+1]
                fineTune = 20
                if abs(start-stop) == 1: #The jumps across the graph have two edges that are one apart
                    if abs(p[stop]-p[start]) < 300 and abs(t[stop]-t[start]) < 300: #there are however parts of the coil which are one
                                                                                    # apart and dont jump across the graph so we should
                                                                                    # still graph those
                        xs = np.linspace(p[start], p[stop], fineTune)
                        ys = np.linspace(t[start], t[stop], fineTune)
                        ax.plot(xs, ys, c)
                        ax.scatter(xs[0], ys[0], s = 15)
                        ax.scatter(xs[1], ys[1], c = "orange", s = 15)
                else: #everything should be graphed with the blue and orange dots to signify direction
                    xs = np.linspace(p[start], p[stop], fineTune)
                    ys = np.linspace(t[start], t[stop], fineTune)
                    ax.plot(xs, ys, c) 
                    ax.scatter(xs[0], ys[0], c = "blue", s = 15)
                    ax.scatter(xs[1], ys[1], c = "orange", s = 15) 
             
            

        elif coiltype[n] == 'toroidal_field':     ax.plot(p, t, c='b')
        elif coiltype[n] == 'Vertical_Field_Coil': two = 1+1 #dont plot the vertical coils
        else: print('COIL-TYPE ERROR!')
        

    plt.title('HIDRA Magnetic Coils')
    plt.xlabel('Computational Toroidal ($\\varphi$) Angle')#Phi Angle Physical (0 is West end, or some periodic multiple)')
    plt.ylabel('Poloidal ($\\theta$) Angle')

    ax.set_xticks(np.linspace(0,360,21))
    ax.set_yticks(np.linspace(-180,180,21))
    ax.vlines([90,180,270], -200, 200, colors= 'k', linestyles="dashed", linewidth= 2)
    ax.hlines(0, -10, 370, colors='k', linestyles='dashed', linewidth=2)
    
    plt.grid(True)
    plt.margins(0.05)
    plt.savefig('HIDRA_coil_schematic2D.png', bbox_inches='tight', dpi=600)
    #plt.show()


if __name__ == '__main__':
    main()