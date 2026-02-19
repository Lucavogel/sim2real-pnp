from omni.isaac.core import World
from omni.isaac.urdf import URDFImporter
import omni.kit.app

# Lancer l'app
app = omni.kit.app.get_app()

# Créer le monde
world = World(stage_units_in_meters=1.0)

# Importer URDF
importer = URDFImporter()
importer.import_urdf(
    urdf_path="/home/ajin/workspace/sim2real-pnp/environ/ur10/urdf_converted/ur10.urdf",
    usd_path="/home/ajin/workspace/sim2real-pnp/environ/ur10/ur10.usd",
    fix_base=True
)

print("✅ URDF importé en USD")

# Laisser le temps à Kit
app.update()
