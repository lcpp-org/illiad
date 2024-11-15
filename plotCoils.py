 ############
 ## IMPORTS
import pandas as pd
import numpy as np

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
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
            rtp = XYZ_to_RTP(np.ascontiguousarray(point), Rmajor)
            r[i] = rtp[0]
            t[i] = np.degrees(rtp[1])
            p[i] = np.degrees(rtp[2])
            #X[i] = (r*np.cos(theta)) * np.cos(phi)
            #Y[i] = (-r*np.cos(theta)) * np.sin(phi)
            #Z[i] = r * np.sin(theta)
        
        if coiltype[n] == 'Helix':
            if float(coilpts[0][3]) < 0:    c = 'r'
            elif float(coilpts[0][3]) > 0:  c = 'g'
            
            pends = np.append(argrelextrema(p, np.greater), argrelextrema(p, np.less))
            tends = np.append(argrelextrema(t, np.greater), argrelextrema(t, np.less))
            
            edges = np.append(pends, tends)
            edges = np.sort(edges)

            for i in range(len(edges)-1):
                start = edges[i]
                stop = edges[i+1]
                ax.plot(p[start:stop], t[start:stop], c)

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
    ax.grid(False)
    plt.margins(0.05)
    plt.legend()
    #plt.savefig('HIDRA_mesh.png', bbox_inches='tight', dpi=600)
    plt.show()


if __name__ == '__main__':
    main()