import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from time import perf_counter
import logging
#import numba as nb
#from numba import jit

## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
#@jit(nb.types.Array(nb.float64, 1, "C")(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def blines(t, p_XYZ, field):
    B = np.zeros(3)
    direction = 1

    B = field.interpField(p_XYZ[:3])

    dY = direction * B / np.linalg.norm(B)

    return dY


##===============##
## DEFINE SOLVER ##
##===============##
def solvePoincare(init_cond, maxLength, field, solver_events):
    log = logging.getLogger()
    log.info(f'Start IC: {init_cond}')

    span = (0.0, maxLength)
    t_startInd = perf_counter()
    fieldlines = solve_ivp(blines, span, init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            #method='RK45', max_step=1e-2, rtol=1e-12, atol=1e-10) #3e-4   # 0
            #method='RK45', rtol=1e-9, atol=1e-9) #3e-4                    # 1
            method='RK45', rtol=1e-10, atol=1e-10) #3e-4                  # 2
            #method='LSODA', rtol=1e-10, atol=1e-10)                       # 3b
            #method='DOP853', rtol=1e-9, atol=1e-9) #3e-4 max_step=1e-2,   # 4
    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd
    if fieldlines.success:
        log.info(f'Solver Success for IC:{init_cond}\n*Took {elapsed_timeInd} sec.')
    else:
        log.critical(f'Solver Failure for IC:{init_cond}')

    data = fieldlines.y_events

    #log.info(f'Wall Events: {fieldlines.y_events[0]}')
    log.info(f'Wall Event at t= {fieldlines.t_events[0]}')
    #print('List of Event 1: ', fieldlines.y_events[1])

    return data