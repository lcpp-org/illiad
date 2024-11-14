import numpy as np
import class_outputHandler as out
import matplotlib.pyplot as plt

def calc_Magnetic_axis(folder, filename):
    Rmajor = 0.72

    simOut = out.IOHandler(folder)
    data = simOut.loadNumpyData(filename)
    Rs = []
    
        

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    index = data.shape[0] -1
    
    magAxis_r = data[index][1]
    
    magAxis_theta = data[index][0][:magAxis_r.size]   
    plt.scatter(magAxis_theta, magAxis_r, marker='.', s=1.5, c='k', linewidths=0.0)
    

    magAxis_x = np.zeros_like(magAxis_r)

    magAxis_y = np.zeros_like(magAxis_r)

    for i, r in enumerate(magAxis_r):
        magAxis_x[i] = r *np.cos(magAxis_theta[i])
        magAxis_y[i] = r * np.sin(magAxis_theta[i])


    mean_x = np.nansum(magAxis_x)/magAxis_x.size
    mean_y = np.nansum(magAxis_y)/magAxis_y.size
    r = np.sqrt(mean_x**2 + mean_y**2)
    theta = np.arctan2(mean_y,mean_x)
    plt.scatter(theta, r, marker = 'o', s = 50, linewidths=0.0)
    plt.show()
    Rs.append([np.sqrt((Rmajor+mean_x)**2 + mean_y**2), Rmajor+mean_x, mean_y])
    Rs = np.array(Rs)
    radii = Rs[:, 0]


    return(Rs[np.argmin(radii)])

print(calc_Magnetic_axis("1deg_1q3_10p_10t_400s", "Poincare_180.npy"))

