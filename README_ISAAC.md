Isaac Sim helper: spawn_ur10.py
=================================

This short README explains how to run `control/spawn_ur10.py` to create a
minimal stage and place a UR10 placeholder prim inside NVIDIA Isaac Sim.

When to use
 - You are running Isaac Sim and want a quick script to create a stage
   with a ground plane and a `/World/ur10e` prim.

How to run
 - From Isaac Sim Script Editor: open `control/spawn_ur10.py` and execute it.
 - From Isaac Sim's bundled Python (example, adapt to your install):

```bash
/path/to/IsaacSim/python.sh /home/ajin/Documents/GitHub/sim2real-pnp/control/spawn_ur10.py
```

 - Optionally pass a USD or URDF path as argument to attempt importing an
   asset into `/World/ur10e`:

```bash
/path/to/IsaacSim/python.sh /home/ajin/Documents/GitHub/sim2real-pnp/control/spawn_ur10.py /path/to/ur10.usd
```

Environment variables
 - `UR10_USD_PATH` or `UR10_URDF_PATH`: set one of these to automatically
   attempt to import or reference an asset when the script runs.

Notes and troubleshooting
 - The script must run inside Isaac Sim's Python. If you run it with
   the system Python you will see import errors for `omni` and `pxr`.
 - Importing URDF/USD depends on the Isaac Sim version and installed
   extension (some versions provide `omni.isaac.urdf.importer`). The
   script attempts a best-effort import and will print actionable
   messages if the importer is not available.
 - If you need a specific UR10 USD/URDF asset and don't have one,
   search the Isaac Sim Asset Browser or convert a URDF/UR5 model to USD.

Next steps you might ask me to do
 - Wire a real UR10 USD asset into the repo (if you provide the file).
 - Implement a richer spawner that sets joint limits, collisions, and a
   dynamics-enabled articulation using the Isaac APIs.
