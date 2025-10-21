#!/usr/bin/env python3
"""
Script de vérification de l'environnement Isaac Lab + ROS2
Vérifie que tous les prérequis sont installés et configurés
"""

import sys
import os
import subprocess
from pathlib import Path

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def check(condition, success_msg, fail_msg):
    """Affiche le résultat d'une vérification"""
    if condition:
        print(f"{Colors.GREEN}✅ {success_msg}{Colors.RESET}")
        return True
    else:
        print(f"{Colors.RED}❌ {fail_msg}{Colors.RESET}")
        return False

def run_command(cmd, capture_output=True):
    """Exécute une commande et retourne le résultat"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=capture_output, 
            text=True, 
            timeout=5
        )
        return result.returncode == 0, result.stdout
    except:
        return False, ""

def main():
    print(f"\n{Colors.BOLD}{Colors.BLUE}=== Vérification Environnement Isaac Lab + ROS2 ==={Colors.RESET}\n")
    
    total_checks = 0
    passed_checks = 0
    
    # 1. Isaac Lab
    print(f"{Colors.BOLD}📦 Isaac Lab{Colors.RESET}")
    isaac_path = Path.home() / "work2" / "IsaacLab"
    if check(
        isaac_path.exists(),
        f"Isaac Lab trouvé: {isaac_path}",
        f"Isaac Lab NON TROUVÉ: {isaac_path}"
    ):
        passed_checks += 1
        
        # Vérifier isaaclab.sh
        isaaclab_sh = isaac_path / "isaaclab.sh"
        if check(
            isaaclab_sh.exists() and os.access(isaaclab_sh, os.X_OK),
            "isaaclab.sh exécutable",
            "isaaclab.sh non exécutable"
        ):
            passed_checks += 1
        total_checks += 1
    total_checks += 1
    
    # 2. ROS2
    print(f"\n{Colors.BOLD}🤖 ROS2{Colors.RESET}")
    success, _ = run_command("which ros2")
    if check(success, "ROS2 installé", "ROS2 NON INSTALLÉ"):
        passed_checks += 1
        
        # Vérifier version
        success, output = run_command("ros2 --version")
        if success and output:
            print(f"   Version: {output.strip()}")
    total_checks += 1
    
    # 3. Workspace ROS2
    print(f"\n{Colors.BOLD}📁 Workspace ROS2{Colors.RESET}")
    workspace_path = Path.home() / "work2" / "ur10"
    if check(
        workspace_path.exists(),
        f"Workspace trouvé: {workspace_path}",
        f"Workspace NON TROUVÉ: {workspace_path}"
    ):
        passed_checks += 1
        
        # Vérifier package ur_coppeliasim
        package_path = workspace_path / "src" / "ur_coppeliasim"
        if check(
            package_path.exists(),
            "Package ur_coppeliasim trouvé",
            "Package ur_coppeliasim NON TROUVÉ"
        ):
            passed_checks += 1
            
            # Vérifier les scripts
            scripts = [
                "isaaclab_bridge.py",
                "simple_isaaclab_test.py",
                "test_isaaclab_moveit.py"
            ]
            all_scripts = True
            for script in scripts:
                script_path = package_path / "scripts" / script
                if not script_path.exists():
                    print(f"   {Colors.YELLOW}⚠️  Script manquant: {script}{Colors.RESET}")
                    all_scripts = False
            
            if check(
                all_scripts,
                "Tous les scripts Isaac Lab présents",
                "Certains scripts manquants"
            ):
                passed_checks += 1
            total_checks += 1
        total_checks += 1
        
        # Vérifier compilation
        install_path = workspace_path / "install" / "ur_coppeliasim"
        if check(
            install_path.exists(),
            "Workspace compilé (install/ur_coppeliasim existe)",
            "Workspace NON COMPILÉ - Exécutez: colcon build"
        ):
            passed_checks += 1
        total_checks += 1
    total_checks += 1
    
    # 4. MoveIt2
    print(f"\n{Colors.BOLD}🎯 MoveIt2{Colors.RESET}")
    success, _ = run_command("ros2 pkg list | grep moveit")
    if check(success, "MoveIt2 installé", "MoveIt2 NON INSTALLÉ - sudo apt install ros-humble-moveit"):
        passed_checks += 1
    total_checks += 1
    
    # 5. NVIDIA
    print(f"\n{Colors.BOLD}🎮 NVIDIA{Colors.RESET}")
    success, output = run_command("nvidia-smi")
    if check(success, "Driver NVIDIA OK", "Driver NVIDIA NON DÉTECTÉ"):
        passed_checks += 1
        # Extraire version
        for line in output.split('\n'):
            if 'Driver Version' in line:
                print(f"   {line.strip()}")
                break
    total_checks += 1
    
    # 6. Python packages
    print(f"\n{Colors.BOLD}🐍 Python Packages{Colors.RESET}")
    packages_to_check = [
        ("torch", "PyTorch (pour Isaac Lab)"),
        ("rclpy", "ROS2 Python client"),
    ]
    
    for package, description in packages_to_check:
        try:
            __import__(package)
            check(True, f"{description} installé", "")
            passed_checks += 1
        except ImportError:
            check(False, "", f"{description} NON INSTALLÉ")
        total_checks += 1
    
    # 7. Display
    print(f"\n{Colors.BOLD}🖥️  Display{Colors.RESET}")
    display = os.environ.get('DISPLAY')
    if check(display is not None, f"DISPLAY configuré: {display}", "DISPLAY non configuré"):
        passed_checks += 1
    total_checks += 1
    
    # Résumé
    print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
    percentage = (passed_checks / total_checks * 100) if total_checks > 0 else 0
    
    if percentage == 100:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ TOUS LES TESTS RÉUSSIS! ({passed_checks}/{total_checks}){Colors.RESET}")
        print(f"\n{Colors.GREEN}Vous êtes prêt à utiliser Isaac Lab + ROS2!{Colors.RESET}")
        print(f"\n{Colors.BOLD}Prochaines étapes:{Colors.RESET}")
        print(f"  1. Testez Isaac Lab seul:")
        print(f"     cd ~/work2/IsaacLab")
        print(f"     ./isaaclab.sh -p scripts/tutorials/00_sim/spawn_prims.py")
        print(f"\n  2. Testez le bridge simple:")
        print(f"     ./isaaclab.sh -p ~/work2/ur10/src/ur_coppeliasim/scripts/simple_isaaclab_test.py")
        print(f"\n  3. Lancez MoveIt2 complet:")
        print(f"     cd ~/work2/ur10 && source install/setup.bash")
        print(f"     ros2 launch ur_coppeliasim ur_isaaclab_moveit.launch.py")
        print(f"\n📚 Voir QUICKSTART_ISAACLAB.md pour plus de détails")
        return 0
    elif percentage >= 70:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  PRESQUE PRÊT ({passed_checks}/{total_checks}){Colors.RESET}")
        print(f"\n{Colors.YELLOW}Corrigez les éléments marqués ❌ ci-dessus{Colors.RESET}")
        return 1
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ CONFIGURATION INCOMPLÈTE ({passed_checks}/{total_checks}){Colors.RESET}")
        print(f"\n{Colors.RED}Plusieurs prérequis manquants. Consultez README_ISAACLAB.md{Colors.RESET}")
        return 2

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Vérification interrompue{Colors.RESET}")
        sys.exit(130)
