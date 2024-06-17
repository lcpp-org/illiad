import numpy as np
import scipy as sp
from scipy.integrate import solve_ivp
from time import perf_counter
import logging

from coordtrans import *

## ==================================== ##-
## FIELD LINE SOLVER (FROM "Bfield.py") ##
## ==================================== ##
def blines(t, p_XYZ, field):
    direction = 1
    B = np.zeros(3)
    B, dum_ = field.interpField(p_XYZ[:3])

    # hard-coded, hacky error field implementation
    B[0] += 0.0002
    B[1] += -0.0002

    dY = direction * B / np.linalg.norm(B)

    return dY


##===============##
## DEFINE SOLVER ##
##===============##
def solvePoincare(particle, field, solver, rtl, atl, solver_events):

    if particle.type == 'ion':
        #print('pos0_XYZ: {}'.format(particle.pos0_XYZ))
        #print('vel0_XYZ: {}'.format(particle.vel0_XYZ))
        init_cond = np.concatenate((particle.pos0_XYZ, particle.vel0_XYZ))
        print('init_cond: {}'.format(init_cond))
    else:
        init_cond = particle.pos0_XYZ

    maxLength = particle.maxLife
    log = logging.getLogger()
    log.info('Start IC: {}'.format(particle.particleID))

    t_startInd = perf_counter()
    #fieldlines = solve_ivp(blines, (0.0, maxLength), init_cond, args = ([field]),
    fieldlines = solve_ivp(particle.pushXYZ, (0.0, maxLength), init_cond, args = ([field]),
            dense_output=False,
            events = solver_events, 
            method=solver, rtol=rtl, atol=atl)
    t_stopInd = perf_counter()
    elapsed_timeInd = t_stopInd - t_startInd

    tmax = np.max(fieldlines.t)

    if fieldlines.status == 0: #solver ran to max. time
        log.info('Success!: Particle {} of {} took {:.4f} sec.\tWall Event at t={}'.format(particle.particleID, particle.particleCount, elapsed_timeInd, fieldlines.t_events[0]))
    elif fieldlines.status == 1: #termination event
        log.info('Success!: Particle {} of {} took {:.4f} sec.\tWall Event at t={}'.format(particle.particleID, particle.particleCount, elapsed_timeInd, fieldlines.t_events[0]))
    else: #solver failure
        log.critical('FAILURE!: Particle {}'.format(particle.particleID))

    data = fieldlines.y_events[:]

    return tmax, data