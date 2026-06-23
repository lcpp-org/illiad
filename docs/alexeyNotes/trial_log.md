# Testing runPoincare.py


## Trial 1: 
**What I tried:**  
Full simulation at iota3 with 1000 spins and 53 lines with LSODA solver. 

**Output file:** `iota3FWD_1000spins_53_Lines_LSODA`

**Notes:** \
My first official run of the full simulation. The plots looked good; 53 lines at 1000 spins took about 253 seconds to generate.  

## Trial 2:
**What I tried:** \
iota3 with 1000 spins and 10 lines and LSODA solver. The purpose is to see the runtime difference. 

**Output file:** `iota3FWD_1000spins_10_Lines_LSODA`

**Notes:** \
Based on the log file, it took 603.89 to finish, which is surprising as it took linger than 53 lines.

## Trial 3: 
**What I tried:** \
iota3 with 1000 spins and 10 lines, trying RK45, which is a faster solver but less accurate than LSODA. Once again, want to see the runtime difference. 

**Output file:** \ `iota3FWD_1000spins_10_Lines_RK45`

**Notes:** \
It took ~250 seconds for solver which less than before but not by a lot. The 600-second spike is weird but is probably because of the node difference. I assume some nodes are faster than other.

## Trial 4:
**What I tried:** \
Full iota4 simulation with 1000 spins and 53 lines (LSODA).

**Output file:**
`iota4FWD_1000spins_53_Lines_LSODA`

**Notes:**\
The graph looks good; however, the islands do not have enough lines inside as iota3. The runtime was ~629 seconds. 

## Trial 5/6/7:
**What I tried:**\
One trial with 10 lines and 1000 spins (LSODA), one trial with 53 lines and 1000 spins (RK45), one trial with 53 lines, 1000 spins (LSODA), and 100 planes. 
The purpose is to see the difference in the runtime and also test that `NPLANES` changes the number of generated graphs as I think it does.

**Output files:**\
`iota4FWD_1000spins_10_Lines_LSODA`, `iota4FWD_1000spins_53_Lines_RK45`,
`iota4FWD_1000spins_53_Lines_LSODA_100Planes`

**Notes:**\
RK45 solver took significantly less time to solve (300 seconds vs avg. 600 seconds), and for some reason, 100 planes took 800 seconds, which is not what I expected. It is probably a good idea to re-run some of these to see if I get the same runtime, and if so, then it is probably just caused by the node that I queue on the cluster. 

## Trial 8/9:
**What I tried:** \
full iota5 with 10 lines vs 53 lines.

**Output files:** \
`iota5FWD_1000spins_10_Lines_LSODA`,
`iota5FWD_1000spins_53_Lines_LSODA`

**Notes:**\
The runtime was almost identical, which is really weird. However, I was not seeing any islands in the 53 line case.

## Trial 10:
**What I tried:** \
Full iota5 with 53 lines and 1000 spins.

**Output file:** \
`iota7FWD_1000spins_53_Lines_LSODA`

**Notes:** \
Not seeing any island, just a really dense surfaces.

# Testing runFluxCalc.py
 **What I tried:** \
 First time running FluxCalc. Tried with iota3 1000 spins and 10 lines (RK45). Tried iota3 1000 spins 53 lines (LSODA). Tried iota3 1000 spins 53 lines (LSODA). Tried iota4 1000 spins 53 lines (RK45)

 **Notes:** \
 All 4 trials took about an hour to finish. Before I was able to get the first job to run, I was testing what `ANLYS_SUBDIR` is. Turns out it was just the name of the subdirectory that is created inside `ANLYS_DIR`. Then for some reason when I wrote the nonexistent name in `ANLYS_DIR` in created a file inside `output` folder and gave an error. Will have to double-check because it only happened once. \ 
 Out of 4 trials, the only "good" plot was `iota3FWD_1000spins_53Lines_LSODA_FLUX` which kind of captured the general shape of surface parameter vs surface index. I think this does make sense because out of 4, this was the only one that had enough surfaces in the islands. 

# Testing runFluxGrad.py
 **What I tried:** \
Tried running `iota4FWD_1000spins_53Lines_LSODA_flux` to see what `runFluxGrad.py` does. 

**Nots:** \
`runFluxGrad.py` is running fine on the local computer, no need for cluster. Graphs look noisy, I believe I might have been inconsistent with `'MAX_SUBSETS'` and other input parameters which led to bad plots.


# Testing runBorisEfieldy.py

**What I tried:** \
First time running BorisEfieldy. Tried running my "best" `iota3FWD_1000spins_53Lines_LSODA_flux` trial. In the first attempt accidently wrote incorrect directory, once again it created empty folder and gave an error. The error said `No such file or directory: '/scratch/basov2/code/fieldlines-uiuc/output/iota3FWD_1000spins_53LinesLSODA_flux/data/Poincare_002.npy'`. It specifically could not find Poincare_002.npy, not sure why it starts with 002 instead of 001.
Second attempt got a new error `KeyError: 'allocated_bytes.all.current'` I think it is because I set `gpus-per-node=0` in the batch file so the code is running on CPU, but still tries to print GPU memory stats.

**Notes:**\
It worked on the third attempt! I got the streaks, everything is very diffused and particles are unconfined. Not a good try, I need cleaner flux.


# Re-running FluxCalc and FluxGrad 6/10/2026
The goal is to get a nicer profile for Boris.

## Trial 1
**What I tried:** \
Tried switching from 20 LCFS to 8 which is the suggestion by the Poincare log file. 

**Output file:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/second_flux_run`

**Notes:**\
Flux_v_Surface plot looked messy between surface index 8–20. Many lines jump around and drop to zero, for one of the angles the toroidal flux has a huge spike near surface index 8 and goes close to 10000 g*m^2. `FluxGrad` gave uniformly dense `LinearFluxNorm` which is not physically correct. Because the Poincare plots look fine, this is probably the issue with the input during flux calc. 


## Trial 2
**What I tried:** \
Chaning index to 10 for LCFS, also changed NPHI from 360 to 90 to hopefully speed up the runtime for testing. Lastly, decreased by tolarances by the order of magnitude.

**Notes:** \
A major improvment, still not perfect but now most of the flux​ curves follow a similar trend. What concerns me is the consistent spike in all trials at around: 24, 32, and 34, HOWEVER, this might actually be a good sign because these might be the islands. Lastly, the runtime did decrease which is very nice, I'm going to keep it this way for now.

## Trial 3
**What I tried:** \
Keeping the same tolarance but now going back to LCFS = 20. 

**Outpuf File:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/third_flux_run`

**Notes:** \
Accidentally override previous subdir, BUT the result is much cleaner. Still seeing the same spikes at same indices. Gonna do another run but in a new non existing subdir.

## Trial 4
**What I tried:** \
'LCFS_INDEX': 20, 'INTEGRATE_EPSABS': 5e-2, 'INTEGRATE_EPSREL': 5e-3. Just rerunning previous trial to keep it organized.

**Outpuf File:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/fourth_flux_run`

**Notes:** \
Very happy with this run, I think this is my best one so far. There is now a real gradient and its no longer just a blob but the edges are noisy. Maybe NPHI was too low or maybe I should try smoothing. I will start with increasing NPHI back to 360.

## Trial 5
**What I tried:** \
Changed NPHI from 90 back to 360. The runtime is back to taking hours....

**Outpuf File:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/sixth_flux_run`

**Notes:** \
Not really an improvment, there is a lot more outliers but the general shape is still being captured. Maybe it is not the NPHI issue. I will go back to NPHI = 90 for testing and play with smoothing. However, I think its the issue with spline not really integration or resolution because at some angles splines do not stay close to the Poincare points. The gradient is gone, this basically means that increasing NPHI in my case just add more noicy data. 

## Trial 6
**What I tried:** \
Back to 90 NPHI and smoothing is set to baseline 1e-6.

**Outpuf File:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/seventh_flux_run`

**Notes:** \
Maybe looks better than NPHI = 360 but no difference in previouse attempts. The density plot looks identical. Need to play with smoothing more, but if that does not work out, replotting Poincare is the way to go.

# Re-running FluxGrad 6/10/2026
## Trial 1
**What I tried:** \
Testing new `input_params['SAVE_BEST_PROFILE']` parameter input.

**Outpuf File:** \
`iota3FWD_1000spins_53Lines_LSODA_flux/fifth_flux_run`

**Notes File:** \
Everything looks good, profile saved inside the subdir.


# Running FluxCalc and FluxGrad iota4 6/10/2026
The goal is to get a nice profile for iota4 that has not surfaces inside the islands
**What I tried:** \
Immediately changed the tolarance to `'INTEGRATE_EPSABS': 5e-2,` and `'INTEGRATE_EPSREL': 5e-3,`. LCFS is set to 4 based on Poincare and NPHI = 90

**Outpuf File:** \
`iota4FWD_1000spins_53Lines_LSODA_flux/first_flux_run`

**Notes:** \
Insane run, plot look amazing, but it is totally random. Nice gradient, super smooth, lack of any noice, looks awesome. It does look kind of layers rather than a continuous gradient, so I think applying smoothing for next trial is a good choice.


# Running iota4 6/13/2026
## Trial 1
Running previous iota4 witn NPHI = 90 for streak did not work, the code gives the error
``` Python
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

## Trial 2
**Notes:** \
Ran the whole simulation for iota4 with NPHI=360, the results showed only tiny difference between NPHI=90 and NPHI=360. For the NPHI=360 case, the `Flux_v_Surface.png` had more noise, while magnitude and density plots were ALMOST identical, NPHI=360 looks a little differnet on the edge but the difference is almost unnoticeable
Additionally, I was able to run 43200 particles on my local computer, I compared the results to the one from the cluster to see if they match. Everything looked good and the runtime was only about 15 minutes. This is really good because the queue time can sometimes take days. The only problem is that simulation crashed while generating `IonTraces.png`. I still gonna figure out how to run NPHI=90.

PC SPECS:
- **Processor:** AMD Ryzen 9 5900HS 
- **RAM:** 24.0 GB 
- **Graphics Card:** NVIDIA GeForce RTX 3070 Laptop (8 GB)

The results showed "concentrated" streaks along the vessel wall, no diffusion. There is more particle deposition at the outer edge which makes sense because this is where particles are spawned.

**Output File:**
`iota4FWD_1000spins_53Lines_LSODA_flux/fullNPHI_flux_run`

## Trial 3
Re-running Boris on the home pc with 1200 tracked particles to see if the simulation will crash again.

**Notes:**\
No crashes! Got the `IonTraces.png`, the plasma looks confined and I can the periodicity

# Running iota3 6/13/2026
Trying to fix previous attempts (fourth) to generate iota3 profile, by adjusting
`SMALLEST_ISLAND_INDEX`

**Notes:**\
"Best" flux profile is still janky no matter what I try to fix. After this I'm just going to re run Poincare one more time.

# Running iota3 with no islands 6/15/2026
## Trial 1
Generated 5 Poincares for different angles with 1200 spins which includes: baseline(theta=180), 90-deg, 135-deg, 160-deg, 170-deg.
Initially I thought that 170-deg looked the "best" because there was no surfaces inside the islands and there was a good amount of surfaces at the magnetic axis. After running `fluxCalc` the "best flux profile" still looked horrible. It is able to map the entire shape from center to LCFS but there is still that tumor looking shape between surface 20 and 30. This exact shape was present in previous iota3 attempt not sure what it is. However, the magnitude and density looked okay. Everything is continuous, density has a gradient, but islands are explicitly shown (unlike iota4 no island case) and they are dense regions, this is not what we want. I think the reason why I got gradient inside islands is because there is still a lot of surfaces inside the islands, I want to re run Poincare for 165 degree to get rid of some surfaces.
Additionally, the edge looks a bit noicy, probably have to play with smoothing a little. Also, maybe the tumor shape on the best flux profile is caused by bad tolarance when integrating, I will play with that as well. 
Lastly, since I was running this trial at NPHI = 90 I could not get Boris to run

**Output File:** \
`iota3_1200spins_53Lines_LSODA_170deg/iota3_test1_170deg`

## Trial 2
Re running 170-deg case for NPHI=180


## Trial 3
Generated 165-deg Poincare 




































