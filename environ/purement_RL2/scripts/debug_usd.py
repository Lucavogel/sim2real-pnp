"""Script pour inspecter la structure de env_v1.usd"""

import argparse
from isaaclab.app import AppLauncher

# Créer l'argument parser
parser = argparse.ArgumentParser(description="Inspect USD structure")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Lancer Isaac Sim
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ===== APRÈS le lancement de l'app, importer omni =====
import omni.usd
from pxr import Usd, UsdGeom

def inspect_usd(usd_path: str):
    """Inspecte la structure d'un fichier USD"""
    print(f"\n{'='*60}")
    print(f"Inspection de : {usd_path}")
    print(f"{'='*60}\n")
    
    # Ouvrir le stage
    stage = Usd.Stage.Open(usd_path)
    
    if not stage:
        print(f"❌ Erreur : Impossible d'ouvrir {usd_path}")
        return
    
    print("=== Structure du USD ===\n")
    
    def print_prim_tree(prim, indent=0):
        """Affiche récursivement la structure"""
        type_name = prim.GetTypeName() if prim.GetTypeName() else "Scope"
        print("  " * indent + f"├─ {prim.GetPath()} ({type_name})")
        
        for child in prim.GetChildren():
            print_prim_tree(child, indent + 1)
    
    root = stage.GetPseudoRoot()
    print_prim_tree(root)
    
    # Chercher spécifiquement les articulations et robots
    print("\n" + "="*60)
    print("=== Articulations et Robots trouvés ===")
    print("="*60 + "\n")
    
    found_robots = []
    found_articulations = []
    
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        
        # Chercher les robots
        if any(keyword in prim_path for keyword in ["Robot", "robot", "UR10", "ur10", "UR", "arm"]):
            found_robots.append(prim_path)
            print(f"🤖 Robot trouvé : {prim_path}")
            print(f"   Type : {prim.GetTypeName()}")
            
            # Afficher les enfants immédiats
            children = list(prim.GetChildren())
            if children:
                print(f"   Enfants ({len(children)}):")
                for child in children[:5]:  # Limiter à 5 premiers
                    print(f"      - {child.GetName()} ({child.GetTypeName()})")
                if len(children) > 5:
                    print(f"      ... et {len(children) - 5} autres")
            print()
        
        # Chercher les articulations (PhysicsArticulationRootAPI)
        if prim.HasAPI("PhysicsArticulationRootAPI"):
            found_articulations.append(prim_path)
            print(f"🔧 Articulation : {prim_path}")
    
    # Résumé
    print("\n" + "="*60)
    print("=== RÉSUMÉ ===")
    print("="*60)
    print(f"\n✓ Robots trouvés : {len(found_robots)}")
    for robot_path in found_robots:
        print(f"  → {robot_path}")
    
    print(f"\n✓ Articulations trouvées : {len(found_articulations)}")
    for artic_path in found_articulations:
        print(f"  → {artic_path}")
    
    if found_robots:
        print(f"\n💡 Utilisez ce chemin dans votre config :")
        print(f"   prim_path=\"{found_robots[0]}\"")
    else:
        print("\n⚠️  Aucun robot trouvé. Vérifiez la structure de votre USD.")

# Chemin vers votre USD
usd_path = "/home/ajin/work2/sim2real-pnp/environ/my_env/source/env_v1.usd"

# Inspecter le USD
inspect_usd(usd_path)

# Fermer l'application
simulation_app.close()