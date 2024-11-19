 ############
 ## IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from coordtrans import XYZ_to_RTP
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
        thiscoil = torch.tensor(coilpts, dtype=torch.float64)
        filament = np.ascontiguousarray(thiscoil.T[:3])
        
        r = np.zeros(filament.shape[1])
        t = np.zeros(filament.shape[1])
        p = np.zeros(filament.shape[1])

        for i,point in enumerate(filament.T):
            radius, theta, phi = XYZ_to_RTP(np.ascontiguousarray(point), Rmajor)

            r[i] = radius
            t[i] = np.degrees(theta)
            p[i] = np.degrees(-phi)+360 # Negative to flip to match the helical coils around HIDRA
            #X[i] = (r*np.cos(theta)) * np.cos(phi)
            #Y[i] = (-r*np.cos(theta)) * np.sin(phi)
            #Z[i] = r * np.sin(theta)
        
        if coiltype[n] == 'Helix':
            if float(coilpts[0][3]) < 0:    c = 'r'; firstTime=0
            elif float(coilpts[0][3]) > 0:  c = 'g'; firstTime=0
            
            highends = np.append(argrelextrema(p, np.greater), argrelextrema(p, np.less))
            #highends = np.add(highends, np.ones(len(highends)))
            lowends = np.append(argrelextrema(t, np.greater), argrelextrema(t, np.less))
            #lowends = np.add(lowends, np.ones(len(lowends)))
            
            edges = np.append(highends, lowends)
            edges = np.sort(edges)
            edges = np.append(0, edges)
            edges = np.append(edges, len(p)-1)
            edges = np.array([int(edge) for edge in edges])
            
            
            for i in range(len(edges)-1):
                start = edges[i]
                stop = edges[i+1]
                fineTune = 20
                if abs(start-stop) == 1: 
                    if abs(p[stop]-p[start]) < 300 and abs(t[stop]-t[start]) < 300:
                        """if firstTime == 0:
                            plt.arrow(p[start],t[start], p[stop]-p[start],t[stop]-t[start], head_width=0.5, head_length=0.1, fc='k', ec='k')
                            firstTime == 1
                        else:"""
                        xs = np.linspace(p[start], p[stop], fineTune)
                        ys = np.linspace(t[start], t[stop], fineTune)
                        ax.plot(xs, ys, c)
                        ax.scatter(xs[0], ys[0], s = 5)
                        ax.scatter(xs[1], ys[1], c = "orange", s = 5)
                else:
                    """if firstTime == 0:
                        plt.arrow(p[start],t[start], p[stop]-p[start],t[stop]-t[start], head_width=0.5, head_length=0.1, fc='k', ec='k')
                        firstTime == 1
                    else:"""
                    xs = np.linspace(p[start], p[stop], fineTune)
                    ys = np.linspace(t[start], t[stop], fineTune)
                    ax.plot(xs, ys, c) 
                    ax.scatter(xs[0], ys[0], c = "blue", s = 15)
                    ax.scatter(xs[1], ys[1], c = "orange", s = 15) 
             
            #ax.plot(p, t, c)

        elif coiltype[n] == 'toroidal_field':     ax.plot(p, t, c='b')
        #elif coiltype[n] == 'Vertical_Field_Coil': ax.plot(p, t, c='orange')
        #else: print('COIL-TYPE ERROR!')
        ax.vlines(180, 0, 360, 'black', linewidth=3)
        

    plt.title('HIDRA Mesh')
    plt.xlabel('Phi Angle')
    plt.ylabel('Theta Angle')

    #ax.set_xticks([])
    #ax.set_yticks([])
    #ax.set_zticks([])
    #plt.axis('off')
    plt.grid(True)
    plt.margins(0.05)
    #plt.savefig('HIDRA_mesh.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()