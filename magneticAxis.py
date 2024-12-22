import numpy as np
import class_outputHandler as out
import matplotlib.pyplot as plt

def calc_Magnetic_axis(folder, filenames):
    Rmajor = 0.72

    simOut = out.IOHandler(folder)

    #create the plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    #ax = fig.add_subplot(111, polar=True)
    ax.set_rlim(0, 0.1)
    

    for num in filenames:
        filename = f"Poincare_{num:03}.npy"
        data = simOut.loadNumpyData(filename)
        Rs = []
    

        #find the smallest circles points
        magAxis_r = data[-1][1]
        
        magAxis_theta = data[-1][0][:magAxis_r.size]   
        #plt.scatter(magAxis_theta, magAxis_r, marker='.', s=1.5, c='k', linewidths=0.0)
    

        magAxis_x = np.zeros_like(magAxis_r)

        magAxis_y = np.zeros_like(magAxis_r)

        for i, r in enumerate(magAxis_r):
            magAxis_x[i] = r *np.cos(magAxis_theta[i])
            magAxis_y[i] = r * np.sin(magAxis_theta[i])


        mean_x = np.nansum(magAxis_x)/magAxis_x.size
        mean_y = np.nansum(magAxis_y)/magAxis_y.size

        r = np.sqrt(mean_x**2 + mean_y**2)
        theta = np.arctan2(mean_y,mean_x)
    
        plt.scatter(theta, r, marker = 'o', s = 10, linewidths=0.0, label=f"{num}$^{{\\circ}}$ | r0={np.sqrt((Rmajor+mean_x)**2 + mean_y**2):.2e} | r,z = {Rmajor+mean_x:.2e},{mean_y:.1e}")
        Rs.append([np.sqrt((Rmajor+mean_x)**2 + mean_y**2), Rmajor+mean_x, mean_y])
        Rs = np.array(Rs)
        radii = Rs[:, 0]
    
    plt.legend(loc = "lower center")
    plt.title(f"Magnetic Axes for 5 angles, {filenames}")
    plt.savefig(f"output/{folder}/magnetic_axes_{filenames[0]:03}")

    return(plt, Rs[np.argmin(radii)])

calc_Magnetic_axis("1deg_1q3_10p_10t_400s", [45, 117, 189, 261, 333])



