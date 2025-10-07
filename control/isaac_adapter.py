"""
Isaac Lab adapter stub for the minimal API used by UR_10_rotation.py.

This file contains a lightweight adapter interface that mimics the few
functions and constants from the CoppeliaSim remote API used in the
project. It's a scaffold to implement a proper Isaac Lab integration.

What to implement for a working port:
 - start/stop connection functions (simxStart, simxFinish)
 - object handle retrieval (simxGetObjectHandle)
 - joint read/write (simxGetJointPosition, simxSetJointTargetPosition)
 - optional convenience calls: simxGetObjects, simxGetIntegerParameter

Notes:
 - Isaac Lab / Isaac Sim uses a different API (Python bindings, or gRPC
   based remote control). Porting requires creating a scene (UR10 model)
   and mapping joint names to handles.
 - This adapter intentionally raises NotImplementedError where behavior
   must be filled by the developer. See README_ISAAC.md for next steps.
"""
import time
import math
import numpy as np

# Re-export constants to match the sim shim expectations
simx_opmode_blocking = 0
simx_opmode_streaming = 1
sim_handle_all = -1
sim_intparam_mouse_x = 0
simx_return_ok = 0


class IsaacConnection:
    """A minimal connection object placeholder."""
    def __init__(self):
        self.connected = True
        self.joint_handles = {}


_conn = None


def simxFinish(clientID):
    global _conn
    _conn = None
    return simx_return_ok


def simxStart(ip, port, waitUntilConnected, doNotReconnectOnceDisconnected, timeOutInMs, commThreadCycleInMs):
    """Establish a fake connection object.

    Replace this with actual Isaac Lab connection code. For unit testing
    you can return any non -1 integer to indicate success.
    """
    global _conn
    _conn = IsaacConnection()
    # Return a fake client ID
    return 1


def simxGetObjects(clientID, option, mode):
    # Minimal stub: return empty list
    return simx_return_ok, []


def simxGetIntegerParameter(clientID, param, mode):
    # Stub: return 0
    return simx_return_ok, 0


def simxGetObjectHandle(clientID, name, mode):
    """Map a UR joint name to a handle (index). Implement mapping here.

    Example: 'UR10_joint1' -> 0
    """
    if _conn is None:
        raise RuntimeError('Isaac adapter not started')
    # Look for a mapping in the connection; otherwise try to parse joint index
    try:
        if name.startswith('UR10_joint'):
            idx = int(name.replace('UR10_joint', '')) - 1
            _conn.joint_handles[name] = idx
            return simx_return_ok, idx
    except Exception:
        pass
    # Not found
    return -1, -1


def simxSetJointTargetPosition(clientID, handle, position, mode):
    """Set a desired joint position. In a full adapter this should send
    a command to the Isaac Sim control interface.
    """
    # For now, store last commanded position in the connection object
    if _conn is None:
        raise RuntimeError('Isaac adapter not started')
    # Store by numeric handle
    if not hasattr(_conn, 'last_cmd'):
        _conn.last_cmd = {}
    _conn.last_cmd[handle] = position
    return simx_return_ok


def simxGetJointPosition(clientID, handle, mode):
    """Return a joint position. Without physics we return 0 or the last
    commanded position if available.
    """
    if _conn is None:
        raise RuntimeError('Isaac adapter not started')
    pos = 0.0
    if hasattr(_conn, 'last_cmd') and handle in _conn.last_cmd:
        pos = _conn.last_cmd[handle]
    return simx_return_ok, float(pos)


if __name__ == '__main__':
    print('isaac_adapter executed as main: this is a stub. Implement Isaac Lab client here.')
