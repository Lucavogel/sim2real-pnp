print("NPZ keys:", data.files)

# essayer plusieurs noms possibles
ee_key_candidates = ["ee_pos", "ee_positions", "eef_pos", "eef_positions", "ee", "eef", "ee_xyz"]
ee_key = next((k for k in ee_key_candidates if k in data.files), None)

if ee_key is None:
    raise KeyError(
        f"Aucune clé EE trouvée. Cherché {ee_key_candidates}. "
        f"Clés dispo: {data.files}"
    )

ee_pos = data[ee_key]
print("Using ee key:", ee_key)
