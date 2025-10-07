"""
Lightweight compatibility shim named `sim` so scripts originally written
for the CoppeliaSim remote API can be pointed to another backend.

Usage:
  - By default this shim will try to import the real CoppeliaSim `sim` module
    (if you have it installed in your Python path).
  - To route calls to an Isaac Lab backend, set the environment variable
    SIM_BACKEND=ISAAC and implement the functions in `isaac_adapter.py`.

This file intentionally provides only the small subset of the API used by
`UR_10_rotation.py`. The functions here will raise a clear error when the
requested backend is not available.
"""
from os import environ
import importlib
import sys

# Short list of constants used by UR_10_rotation.py
simx_opmode_blocking = 0
simx_opmode_streaming = 1
sim_handle_all = -1
sim_intparam_mouse_x = 0
simx_return_ok = 0


def _load_backend():
    """Return a module providing a Coppelia-like API.
    Priority:
      1) If SIM_BACKEND=ISAAC -> import control.isaac_adapter
      2) Try to import an actual CoppeliaSim `sim` if available
      3) Fall back to a helpful NotImplemented stub module
    """
    backend = environ.get('SIM_BACKEND', '').upper()
    if backend == 'ISAAC':
        try:
            return importlib.import_module('control.isaac_adapter')
        except Exception as e:
            raise ImportError('Requested SIM_BACKEND=ISAAC but failed to import control.isaac_adapter: ' + str(e))

    # Try to import real CoppeliaSim remote API (if user has it)
    try:
        real_sim = importlib.import_module('sim')
        # If we're running inside a package where this file is visible as control.sim,
        # avoid importing itself recursively
        if real_sim is sys.modules.get(__name__):
            raise ImportError('Imported sim is this shim; skip')
        return real_sim
    except Exception:
        # Return a stub module object implemented below
        return _StubSim()


class _StubSim:
    """Fallback stub that raises actionable errors for every used function."""
    simx_opmode_blocking = simx_opmode_blocking
    simx_opmode_streaming = simx_opmode_streaming
    sim_handle_all = sim_handle_all
    sim_intparam_mouse_x = sim_intparam_mouse_x
    simx_return_ok = simx_return_ok

    def simxFinish(self, clientID):
        raise NotImplementedError('No simulation backend available. Install CoppeliaSim remote API or set SIM_BACKEND=ISAAC and implement control.isaac_adapter')

    def simxStart(self, ip, port, waitUntilConnected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs):
        raise NotImplementedError('No simulation backend available. Install CoppeliaSim remote API or set SIM_BACKEND=ISAAC and implement control.isaac_adapter')

    def simxGetObjects(self, clientID, option, mode):
        raise NotImplementedError('No simulation backend available.')

    def simxGetIntegerParameter(self, clientID, param, mode):
        raise NotImplementedError('No simulation backend available.')

    def simxGetObjectHandle(self, clientID, name, mode):
        raise NotImplementedError('No simulation backend available.')

    def simxSetJointTargetPosition(self, clientID, handle, position, mode):
        raise NotImplementedError('No simulation backend available.')

    def simxGetJointPosition(self, clientID, handle, mode):
        raise NotImplementedError('No simulation backend available.')


# Provide a module-like API by delegating to the selected backend
_backend = _load_backend()

def __getattr__(name):
    # Delegate attribute access to the backend module/object
    return getattr(_backend, name)
