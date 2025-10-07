"""
spawn_ur10.py

Script to create a minimal Isaac Sim stage and place a UR10 prim placeholder.

Usage:
 - Open Isaac Sim, open the Script Editor and run:
     exec(open("/path/to/control/spawn_ur10.py").read())

 - Or from Isaac Sim's bundled Python:
     /path/to/IsaacSim/python.sh /path/to/control/spawn_ur10.py

Behavior:
 - Creates a new stage if none is open.
 - Adds a ground plane, a default camera, and an empty prim at `/World/ur10e`.
 - If you set the environment variable `UR10_USD_PATH` or pass a path to
   a USD/URDF asset, the script will attempt to reference/import it (best
   effort — exact importer availability depends on your Isaac Sim version).

Notes:
 - This script is written to be safe to run inside Isaac Sim. It will
   print clear messages when features (like URDF/USD importers) are not
   available in the running Isaac Sim installation.

"""
import os
import sys

def main(asset_path=None):
    try:
        import omni.usd
        import omni.kit.commands
        from pxr import Usd, Sdf, UsdGeom, Gf
    except ModuleNotFoundError as e:
        print("Erreur : ce script doit être exécuté dans l'environnement Python d'Isaac Sim.")
        print("Détails :", e)
        sys.exit(1)

    ctx = omni.usd.get_context()
    stage = ctx.get_stage()
    if stage is None:
        print("Aucun stage ouvert — création d'un nouveau stage...")
        ctx.new_stage()
        stage = ctx.get_stage()
    else:
        print("Stage existant détecté :", stage.GetRootLayer().identifier)

    # Ensure /World exists
    world_path = '/World'
    if not stage.GetPrimAtPath(world_path):
        world = UsdGeom.Xform.Define(stage, world_path)
        world.GetPrim().SetMetadata('kind', 'assembly')
        print("Prim /World créé")

    # Create a simple ground plane
    ground_path = '/World/Ground'
    if not stage.GetPrimAtPath(ground_path):
        ground = UsdGeom.Mesh.Define(stage, ground_path)
        # A very small plane as placeholder
        points = [Gf.Vec3f(-5.0, 0.0, -5.0), Gf.Vec3f(5.0, 0.0, -5.0), Gf.Vec3f(5.0, 0.0, 5.0), Gf.Vec3f(-5.0, 0.0, 5.0)]
        faceVertexCounts = [4]
        faceVertexIndices = [0, 1, 2, 3]
        ground.CreatePointsAttr(points)
        ground.CreateFaceVertexCountsAttr(faceVertexCounts)
        ground.CreateFaceVertexIndicesAttr(faceVertexIndices)
        print("Ground plane créé à", ground_path)

    # Create a placeholder UR10 prim (empty Xform)
    ur_path = '/World/ur10e'
    if not stage.GetPrimAtPath(ur_path):
        ur = UsdGeom.Xform.Define(stage, ur_path)
        print("Prim placeholder UR10 créée à", ur_path)
    else:
        print("Prim", ur_path, "existe déjà")

    # Attempt to reference or import a provided asset if requested
    chosen = asset_path or os.environ.get('UR10_USD_PATH') or os.environ.get('UR10_URDF_PATH')
    if chosen:
        print("Chemin d'asset fourni:", chosen)
        # If it looks like a USD, create a reference under /World/ur10e
        if chosen.endswith('.usd') or chosen.endswith('.usda') or chosen.endswith('.usdc'):
            try:
                print("Tentative de référence USD depuis:", chosen)
                prim = stage.GetPrimAtPath(ur_path)
                if prim:
                    # Add a reference to the prim (best-effort)
                    prim.GetReferences().AddReference(chosen)
                    print("Référence USD ajoutée à", ur_path)
                else:
                    print("Prim", ur_path, "introuvable pour ajouter la référence")
            except Exception as e:
                print("Échec de l'ajout de la référence USD:", e)
        else:
            # Try to use available URDF importer from omni.isaac if present
            try:
                # Importer API can vary by Isaac Sim version. Try common locations.
                importer = None
                try:
                    from omni.isaac.urdf import import_usd
                    importer = import_usd
                except Exception:
                    try:
                        # older/newer versions
                        from omni.isaac.urdf.importer import import_usd
                        importer = import_usd
                    except Exception:
                        importer = None

                if importer is not None:
                    print("Importeur URDF trouvé — tentative d'import de:", chosen)
                    # The exact function signature varies; call guarded
                    try:
                        # many variants return created prim path or stage
                        importer(chosen, prim_path=ur_path)
                        print("Import URDF tenté — regarder la scène pour vérifier")
                    except TypeError:
                        # fallback: call with single argument
                        importer(chosen)
                        print("Import URDF (fallback) tenté — regarder la scène pour vérifier")
                else:
                    print("Aucun importeur URDF disponible dans cette installation d'Isaac Sim.")
                    print("Place ton USD/URDF dans la variable d'environnement UR10_USD_PATH et réessaie, ou ouvre l'asset dans l'Asset Browser.")
            except Exception as e:
                print("Erreur lors de la tentative d'import URDF/USD:", e)

    print("Script terminé. Vérifie la scène dans Isaac Sim (Stage et Outliner).")


if __name__ == '__main__':
    # Accept an optional path passed as first CLI argument when run with Isaac's python
    path = sys.argv[1] if len(sys.argv) > 1 else None
    main(path)
