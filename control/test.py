# Pour exécuter depuis Isaac Sim Script Editor:
# exec(open("/home/ajin/Documents/GitHub/sim2real-pnp/control/test.py").read())

try:
    import omni.usd
    from omni.isaac.dynamic_control import _dynamic_control
except ModuleNotFoundError:
    print("Erreur : module 'omni' introuvable. Ce script doit être exécuté depuis l'environnement Python d'Isaac Sim.")
    raise SystemExit(1)

dc = _dynamic_control.acquire_dynamic_control_interface()

# Lister toutes les articulations disponibles dans la scène
stage = omni.usd.get_context().get_stage()
print("\n=== Articulations disponibles ===")
for prim in stage.Traverse():
    if prim.IsA(omni.isaac.core.utils.prims.get_prim_type("PhysicsArticulationRootAPI")):
        print(f"Articulation trouvée : {prim.GetPath()}")
        
        # Essayer de récupérer l'articulation via Dynamic Control
        art = dc.get_articulation(str(prim.GetPath()))
        if art:
            nj = dc.get_articulation_joint_count(art)
            print(f"  → {nj} joints détectés")
            for i in range(nj):
                j = dc.get_articulation_joint(art, i)
                # Lister les propriétés du joint
                print(f"    Joint {i}: handle={j}")
                # Essayer d'obtenir le chemin si l'API l'expose
                if hasattr(dc, "get_joint_name"):
                    name = dc.get_joint_name(j)
                    print(f"      nom={name}")

# Essayer avec le chemin racine de l'articulation
print("\n=== Test avec /ur10e ===")
art = dc.get_articulation("/ur10e")
if art:
    nj = dc.get_articulation_joint_count(art)
    print(f"Articulation /ur10e trouvée avec {nj} joints")
    for i in range(nj):
        j = dc.get_articulation_joint(art, i)
        print(f"  Joint index {i}: handle={j}")
        # Tester lecture position
        try:
            pos_array = dc.get_articulation_joint_positions(art)
            if pos_array and len(pos_array) > i:
                print(f"    position={pos_array[i]} rad")
        except Exception as e:
            print(f"    Erreur lecture position: {e}")
else:
    print("Articulation /ur10e non trouvée")
    
print("\n=== Chemins alternatifs à tester ===")
test_paths = [
    "/World/ur10e",
    "/ur10e",
    "/ur10e/ee_joint",
    "/World/ur10e/ee_joint"
]
for path in test_paths:
    j = dc.get_joint(path)
    if j and not (isinstance(j, int) and j < 0):
        print(f"✓ Joint trouvé : {path}")
    else:
        print(f"✗ Joint non trouvé : {path}")