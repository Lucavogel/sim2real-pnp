import torch

paths = [
    "/home/ajin/workspace/sim2real-pnp/Real_robot/validation_mujoco/policy(1).pt",
    "/home/ajin/workspace/sim2real-pnp/Real_robot/model_converged.pt"
]

for p in paths:
    print(f"\n--- Checking {p} ---")
    try:
        data = torch.load(p, map_location="cpu", weights_only=False)
        if isinstance(data, dict):
            print("Keys:", data.keys())
            if "model_state_dict" in data:
                 print("Model State Dict Keys (first 5):", list(data["model_state_dict"].keys())[:5])
            if "optimizer_state_dict" in data:
                 print("Optimizer present")
            # Check for running_mean_std
            if "running_mean_std" in data: # SB3 style?
                 print("Found running_mean_std!")
        else:
            print("Type:", type(data))
            # If it's a JIT script/module
            if isinstance(data, torch.jit.ScriptModule):
                print("Is JIT ScriptModule")
                # Try to list parameters
                print("Named params:", [n for n, _ in data.named_parameters()])
    except Exception as e:
        print(f"Error loading: {e}")
