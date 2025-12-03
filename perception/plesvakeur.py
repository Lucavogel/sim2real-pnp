import pickle

with open("camera_calibration.pkl", "rb") as f:
    data = pickle.load(f)
print(data.keys())
print(data["camera_matrix"])