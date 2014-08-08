import os
import time
from pprint import pformat
import simtk.openmm as mm
from simtk.openmm import app
from simtk.unit import *


class NesterovMinimizer(BaseMinimizer):
    """Local energy minimzation with Nesterov's accelerated gradient descent
    
    Parameters
    ----------
    system : mm.System
        The OpenMM system to minimize
    initialPositions : 2d array
        The positions to start from
    numIterations : int
        The number of iterations of minimization to run
    stepSize : int
        The step size. This isn't in time units.
    """
    def __init__(self, system, initialPositions, numIterations=1000, stepSize=1e-6):
        self.numIterations = numIterations

        integrator = mm.CustomIntegrator(stepSize)
        integrator.addGlobalVariable('a_cur', 0)
        integrator.addGlobalVariable('a_old', 0)
        integrator.addPerDofVariable('y_cur', 0)
        integrator.addPerDofVariable('y_old', 0)
        integrator.addComputeGlobal('a_cur', '0.5*(1+sqrt(1+(4*a_old*a_old)))')
        integrator.addComputeGlobal('a_old', 'a_cur')
        integrator.addComputePerDof('y_cur', 'x + dt*f')
        integrator.addComputePerDof('y_old', 'y_cur')
        integrator.addComputePerDof('x', 'y_cur + (a_old - 1) / a_cur * (y_cur - y_old)')

        self.context = mm.Context(system, integrator)
        self.context.setPositions(initialPositions)

    def minimize(self):
        self.context.getIntegrator().step(self.numIterations)
