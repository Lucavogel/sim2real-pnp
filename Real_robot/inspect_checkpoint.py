import torch

p = "/home/ajin/workspace/sim2real-pnp/Real_robot/model_converged.pt"
print(f"\n--- Deep Check {p} ---")
data = torch.load(p, map_location="cpu", weights_only=False)
state_dict = data["model_state_dict"]
print("All Keys in model_state_dict:")
for k in state_dict.keys():
    print(k)
