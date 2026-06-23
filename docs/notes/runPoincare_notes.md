## Input Section

The initial section lists the HIDRA current table, which is really helpful. However, the explanation of `NTHREADS` and `DOUBLE_LINE` feels out of context because those variables have not been introduced yet.

I think it would make sense to move both down to the input section as: 

```Python
# DEFINE SOLVER PARAMETERS #
SOLVER = "LSODA"#"RK45"#
RTOL = 2.49e-12
ATOL = 1e-8

##  NTHREADS:
#   n > 0: use n threads
#   n = 0: use all available threads
#   n < 0: use all but the last n threads
NTHREADS = -1

# DOUBLE_LINE:
#   True  : trace each field line in both +B and -B directions from the initial position
#           Only use when NTHREADS > NLINES.
#   False : trace each field line only in the +B direction
DOUBLE_LINE = False
```

On the side note, line 99 appears to be redundant:

```Python
    ic_radii = np.array(np.linspace(START_RADIUS, END_RADIUS, NLINES))
```

`np.linspace` is already a numpy array, so making turning it into `np.array` does nothing.

I tried adding resolution into the input parameter and found this part of code inside `poincare.py`:

```Python
# attach plotting function to class instance
for name in dir(plotFuncs):
    func = getattr(plotFuncs, name)
    if callable(func) and not name.startswith("__"):
        if name.startswith("global_"):
            new_name = name.replace("global_", "")  # Remove prefix
        elif name.startswith("poincare_"):
            new_name = name.replace("poincare_", "")  # Remove prefix
            setattr(self, new_name, func)  # Attach to the instance with the new name

```

It took me a while to undestand what is going on in this part and why it is used. From my understanding this stores `poincare_plotPoincareBW` as `self.plotPoincareBW`. of Poincare object. This makes sense and I assume it is future proof for more `poincare_` functions but maybe it would be better to directly set them as

``` Python
self.plotPoincareBW = plotFuncs.poincare_plotPoincareBW

```
 inside `__init__` or I guess the ideal case to put it directly inside Poincare class unless it is used somewhere else, which I have not found. In my opition, this just makes the code cleaner and easier to read.

 