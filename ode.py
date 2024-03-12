import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
#import numba as nb
#from numba import jit

## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
#@jit(nb.types.Array(nb.float64, 1, "C")(nb.float64, nb.types.Array(nb.float64, 1, "C")), nopython=True)
def blines(t, p_XYZ, field):
    B = np.zeros(3)
    direction= -1

    B = field.interpField(p_XYZ[:3])

    dY = direction * B/np.linalg.norm(B)

    return dY


##===============##
## DEFINE SOLVER ##
##===============##
def solvePoincare(init_cond, lineLength, field, solver_events):
    print('IC: ', init_cond)

    span = (0.0, lineLength)
    fieldlines = solve_ivp(blines, span, init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            #method='LSODA', rtol=1e-12, atol=1e-7) 
            #method='DOP853', max_step=1e-2, rtol=1e-12, atol=1e-10) #3e-4 
            method='RK45', max_step=1e-2, rtol=1e-12, atol=1e-10) #3e-4 

    if fieldlines.success:
        print('\nSolver Success for IC: ', init_cond)
    else:
        print('\nSolver Failure for IC: ', init_cond)

    data = fieldlines.y_events

    print('List of Wall Events: ', fieldlines.y_events[0])
    #print('List of Event 1: ', fieldlines.y_events[1])

    return data