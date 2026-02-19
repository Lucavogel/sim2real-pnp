#!/bin/bash
# Script pour convertir l'URDF UR10 ROS2 en USD pour Isaac Lab

set -e  # Arrêter si erreur

echo "======================================================================"
echo "CONVERSION URDF → USD pour UR10"
echo "======================================================================"

# Chemins
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
URDF_PKG="$SCRIPT_DIR/src/Universal_Robots_ROS2_Description"
OUTPUT_DIR="$SCRIPT_DIR/urdf_converted"
URDF_FILE="$OUTPUT_DIR/ur10.urdf"
USD_FILE="$OUTPUT_DIR/ur10.usd"

# Créer dossier output
mkdir -p "$OUTPUT_DIR"

echo ""
echo "📁 Dossier de sortie: $OUTPUT_DIR"
echo ""

# ============================================================
# ÉTAPE 1: Générer le URDF depuis xacro
# ============================================================
echo "🔧 ÉTAPE 1/3: Génération URDF depuis xacro..."
echo "   Source: $URDF_PKG/urdf/ur.urdf.xacro"
echo "   Output: $URDF_FILE"
echo ""

# Source ROS2 workspace
if [ -f "$SCRIPT_DIR/install/setup.bash" ]; then
    source "$SCRIPT_DIR/install/setup.bash"
    echo "   ✅ Workspace ROS2 sourcé"
else
    echo "   ⚠️  install/setup.bash non trouvé, xacro pourrait échouer"
fi

# Générer URDF avec xacro (spécifier ur_type=ur10)
ros2 run xacro xacro "$URDF_PKG/urdf/ur.urdf.xacro" \
    name:=ur10 \
    ur_type:=ur10 \
    tf_prefix:="" \
    joint_limit_params:="$URDF_PKG/config/ur10/joint_limits.yaml" \
    kinematics_params:="$URDF_PKG/config/ur10/default_kinematics.yaml" \
    physical_params:="$URDF_PKG/config/ur10/physical_parameters.yaml" \
    visual_params:="$URDF_PKG/config/ur10/visual_parameters.yaml" \
    > "$URDF_FILE"

if [ -f "$URDF_FILE" ]; then
    echo "   ✅ URDF généré: $(wc -l < "$URDF_FILE") lignes"
else
    echo "   ❌ ERREUR: URDF non généré!"
    exit 1
fi

# ============================================================
# ÉTAPE 2: Convertir URDF → USD avec Isaac Sim
# ============================================================
echo ""
echo "🔄 ÉTAPE 2/3: Conversion URDF → USD avec Isaac Sim..."
echo "   Input:  $URDF_FILE"
echo "   Output: $USD_FILE"
echo ""

# Créer script Python temporaire pour la conversion
CONVERT_SCRIPT="$OUTPUT_DIR/convert_to_usd.py"

cat > "$CONVERT_SCRIPT" <<'EOF'
#!/usr/bin/env python3
"""Script pour convertir URDF en USD avec Isaac Sim"""

import sys
import os
from pathlib import Path

# Importer Isaac Sim
from omni.isaac.kit import SimulationApp

# Lancer Isaac Sim en mode headless
simulation_app = SimulationApp({"headless": True})

# Importer les modules Isaac après lancement
from omni.isaac.core.utils.extensions import enable_extension
enable_extension("omni.isaac.urdf")

import omni.kit.commands
from pxr import Usd, UsdGeom

def convert_urdf_to_usd(urdf_path: str, usd_path: str):
    """Convertit URDF en USD"""
    print(f"📂 Chargement URDF: {urdf_path}")
    
    # Importer URDF
    omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=urdf_path,
        import_config=omni.isaac.urdf.ImportConfig(
            set_default_prim=True,
            create_physics_scene=False,
            import_inertia_tensor=True,
            fix_base=False,
        ),
        dest_path="/World/ur10"
    )
    
    print(f"✅ URDF importé dans la scène")
    
    # Sauvegarder USD
    stage = omni.usd.get_context().get_stage()
    stage.Export(usd_path)
    
    print(f"💾 USD sauvegardé: {usd_path}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_to_usd.py <urdf_path> <usd_path>")
        sys.exit(1)
    
    urdf_path = sys.argv[1]
    usd_path = sys.argv[2]
    
    try:
        convert_urdf_to_usd(urdf_path, usd_path)
        print("\n✅ CONVERSION RÉUSSIE!")
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        simulation_app.close()
EOF

chmod +x "$CONVERT_SCRIPT"

# Lancer la conversion avec Isaac Sim
echo "   🚀 Lancement Isaac Sim pour conversion..."

# Trouver Isaac Sim (chercher dans les emplacements courants)
ISAAC_SIM_PYTHON=""

if [ -f "$HOME/.local/share/ov/pkg/isaac-sim-*/python.sh" ]; then
    ISAAC_SIM_PYTHON=$(ls -t "$HOME/.local/share/ov/pkg/isaac-sim-"*/python.sh 2>/dev/null | head -n1)
elif [ -f "/isaac-sim/python.sh" ]; then
    ISAAC_SIM_PYTHON="/isaac-sim/python.sh"
fi

if [ -z "$ISAAC_SIM_PYTHON" ]; then
    echo ""
    echo "   ⚠️  Isaac Sim non trouvé automatiquement."
    echo ""
    echo "   📝 OPTION MANUELLE:"
    echo "      Lancez Isaac Sim et utilisez le URDF Importer GUI:"
    echo "      Isaac Sim > File > Import > URDF"
    echo "      Fichier: $URDF_FILE"
    echo "      Sauvegardez en: $USD_FILE"
    echo ""
    echo "   Ou exécutez ce script Python avec Isaac Sim:"
    echo "      /path/to/isaac-sim/python.sh $CONVERT_SCRIPT $URDF_FILE $USD_FILE"
    echo ""
    exit 0
else
    echo "   ✅ Isaac Sim trouvé: $ISAAC_SIM_PYTHON"
    "$ISAAC_SIM_PYTHON" "$CONVERT_SCRIPT" "$URDF_FILE" "$USD_FILE"
fi

# ============================================================
# ÉTAPE 3: Vérifier le résultat
# ============================================================
echo ""
echo "🔍 ÉTAPE 3/3: Vérification..."

if [ -f "$USD_FILE" ]; then
    USD_SIZE=$(du -h "$USD_FILE" | cut -f1)
    echo "   ✅ USD créé: $USD_FILE ($USD_SIZE)"
    echo ""
    echo "======================================================================"
    echo "✅ CONVERSION TERMINÉE!"
    echo "======================================================================"
    echo ""
    echo "📂 Fichiers générés:"
    echo "   URDF: $URDF_FILE"
    echo "   USD:  $USD_FILE"
    echo ""
    echo "📝 PROCHAINE ÉTAPE:"
    echo "   Modifiez votre script Isaac Lab pour utiliser ce USD:"
    echo ""
    echo "   from isaaclab.assets import Articulation, ArticulationCfg"
    echo "   "
    echo "   ur10_cfg = ArticulationCfg("
    echo "       prim_path=\"/World/UR10\","
    echo "       spawn=sim_utils.UsdFileCfg("
    echo "           usd_path=\"$USD_FILE\""
    echo "       )"
    echo "   )"
    echo ""
else
    echo "   ❌ USD non créé!"
    echo ""
    echo "   Convertissez manuellement avec Isaac Sim:"
    echo "      File > Import > URDF"
    echo "      Input:  $URDF_FILE"
    echo "      Output: $USD_FILE"
    echo ""
fi
