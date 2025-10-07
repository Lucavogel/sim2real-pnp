# Pour exécuter depuis Isaac Sim Script Editor:
# exec(open("/home/ajin/Documents/GitHub/sim2real-pnp/control/test.py").read())

try:
    import omni.usd
    from omni.isaac.dynamic_control import _dynamic_control
except ModuleNotFoundError:
    print("Erreur : module 'omni' introuvable. Ce script doit être exécuté depuis l'environnement Python d'Isaac Sim.")
    raise SystemExit(1)

JOINT_PATH = "/ur10e/joints/ee_joint"

dc = _dynamic_control.acquire_dynamic_control_interface()

# récupérer le handle du joint (retourne -1 ou None si non trouvé selon version)
try:
    joint = dc.get_joint(JOINT_PATH)
except Exception:
    # certaines versions utilisent None/-1 ou l'API d'articulation — test basique
    joint = dc.get_joint(JOINT_PATH) if hasattr(dc, "get_joint") else None

if joint is None or (isinstance(joint, int) and joint < 0):
    raise RuntimeError(f"Joint non trouvé : {JOINT_PATH}")

# lire position et vitesse actuelles en utilisant le handle 'joint' (évite de rappeler dc.get_joint)
if hasattr(dc, "get_joint_position"):
    pos = dc.get_joint_position(joint)
    vel = dc.get_joint_velocity(joint) if hasattr(dc, "get_joint_velocity") else None
else:
    # fallback pour versions où on lit par articulation/index
    art = None
    try:
        art = dc.get_articulation("/" + JOINT_PATH.split("/")[1]) if hasattr(dc, "get_articulation") else None
    except Exception:
        art = None
    if art is None:
        raise RuntimeError("Impossible de lire les positions : API Dynamic Control différente. Indique la version d'Isaac Sim.")
    # chercher l'index du joint dans l'articulation
    nj = dc.get_articulation_joint_count(art)
    idx = None
    for i in range(nj):
        j = dc.get_articulation_joint(art, i)
        try:
            if hasattr(dc, "get_joint_path") and dc.get_joint_path(j) == JOINT_PATH:
                idx = i
                joint = j
                break
        except Exception:
            pass
    if idx is None:
        raise RuntimeError(f"Index du joint introuvable pour {JOINT_PATH}")
    pos_list = dc.get_joint_positions(art) if hasattr(dc, "get_joint_positions") else None
    vel_list = dc.get_joint_velocities(art) if hasattr(dc, "get_joint_velocities") else None
    pos = pos_list[idx] if pos_list is not None else None
    vel = vel_list[idx] if vel_list is not None else None

print(f"Joint {JOINT_PATH} : position={pos} rad, vitesse={vel} rad/s")

# --- Exemple 1 : position target (utiliser les drives / position controller) ---
target_position = 0.5  # radians (adapter)
# Assure que l'articulation utilise un drive positionnable (stiffness/damping selon besoin)
dc.set_joint_position_target(joint, float(target_position))
# Optionnel : régler gains si l'API expose set_joint_drive / set_joint_stiffness
# dc.set_joint_drive_stiffness(joint, 100.0)
# dc.set_joint_drive_damping(joint, 10.0)

# --- Exemple 2 : appliquer un couple (effort) direct ---
tau = 1.0  # Nm (adapter)
# Attention : appliquer des efforts fonctionne mieux en mode contrôleur bas-niveau
dc.apply_joint_effort(joint, float(tau))

# Remarques :
# - Si la physique tourne en parallèle, exécutez ces appels dans la boucle de simulation
#   ou via un callback (tick) pour qu'ils soient pris en compte.
# - Éviter de "teleporter" un prim dynamique sans désactiver la physique.
# - En cas d'erreur d'API (méthode introuvable), signalez la version d'Isaac Sim et
#   je fournis la variante adaptée.