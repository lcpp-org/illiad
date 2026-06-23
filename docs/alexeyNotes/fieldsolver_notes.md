# Code Notes: runFieldsolver.py

This markdown summarizes the questions and suggested comments that came up while learning the code.

## 1. "READ COIL INPUT FILE" Section - Line 130

The code states that rows are irregularly sized but does not explicitly show the format of the input file and how irregularity is used. I think adding a short example would help readability, especially witu units/meaning of each column.

```Python
## Expected input file structure
## Regular coil row:
## Column 0/1/2 = x, y, z describing the coil geometry
## Column 3     = signed number turns in the coil
## Column 5     = coil type 

## Ending delimiter row:
## Column 0/1/2 = x, y, z describing the coil geometry
## Column 3     = 0.0
## Column 4     = NaN
## Column 5     = coil type
```

In addition, I think it would also be helpful to document the expected units/meaning of each column, especially column 3. The code treates it as the number of `turns`, but some values in the input file are negative.\
Variable `turns` is later used in lines 105-107 and negative sign was never dropped:
```Python
current = turns[n] * I_heli
current = turns[n] * I_toro
current = turns[n] * I_vert
```
I assume this includes the direction of the current flow.

## 2. "DEFINE MESH PERIODICITY" Section - Line 153

The current code is structured as:
```Python
RMAJOR = 0.72 #[m]
RMINOR = 0.19 #[m]

## DEFINE MESH PERIODICITY
## 0: NOT PERIODIC
## 1: 2PI PERIODIC
## >1: HIGHER PERIODICITY (i.e (2PI)/N  PERIODIC)
mesh_periodicity = [ 0, 1, 5]

nr     = int( mesh_size[0] / max(1, mesh_periodicity[0]) )
ntheta = int( mesh_size[1] / max(1, mesh_periodicity[1]) )
nphi   = int( mesh_size[2] / max(1, mesh_periodicity[2]) )

r_prd, theta_prd, phi_prd = mesh_periodicity
```

Overall this makes sense, I didn't have a lot of question about this section. However, earlier input section had already defines `mesh_size` using provided mesh resolutions (`rough, low_res, hi_res`)\
Since `mesh_periodicity` is also a parameter that is part of mesh setup, maybe it would be clearer to place it near the
```Python
## DEFINE MESH RESOLUTION
```
section (line 18). 
Lastly, the code is not fully explicit about the definitions of `theta` and `phi`. Looking deeper into the code shows that `phi` is the toroidal angle and `theta` is poloidal when doing transformation and also the Article provides the angle convention. I think this is probably just a skill issue on my side because after googling, phi is universaly defined as a toroidal angle in torus. I will still leave this note here.

## 3. "loop_through_coils" Section - Line 111

The current code is:
```Python
coilpts = np.asarray(coil, dtype=np.float64)
thiscoil = torch.tensor(coilpts) #, dtype=torch.float64)
filament = thiscoil.T[:3].to(device)
thiscoil = torch.tensor(coilpts, dtype=torch.float64, device=device)
## Mesh-ified
N = filament.shape[1]
Bxyz += biotsavart_mesh(xyz_mesh, filament, current, N)
```
I believe the second assigment to variable `thiscoil` is never used. The tensor that is passed in `biotsavart_mesh` is `filament`. I tested commenting out second `thiscoil` and the code still ran.

## 4. Summary
1. Short comment or example explaining the expected coil input file structure
2. Maybe move machine parameters and `mesh_periodicity` to the `USER INPUTS` section.
3. Check whether line 114 is needed.









