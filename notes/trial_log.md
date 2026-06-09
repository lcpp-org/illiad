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
Tried running `iota4FWD_1000spins_53Lines_LSODA_FLUX` to see what `runFluxGrad.py` does. 

**Nots:** \
`runFluxGrad.py` is running fine on the local computer, no need for cluster. Graphs look noisy, I believe I might have been inconsistent with `'MAX_SUBSETS'` and other input parameters which led to bad plots.

























