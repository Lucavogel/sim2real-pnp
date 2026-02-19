#!/usr/bin/env python3
"""
Script pour vérifier la qualité d'un dataset de trajectoires MoveIt + positions cartésiennes (x, y, z)
"""

import numpy as np
import argparse
from pathlib import Path

def check_dataset(dataset_path):
    """Vérifie un dataset .npz"""
    
    print("\n" + "="*70)
    print("🔍 ANALYSE DU DATASET")
    print("="*70)
    print(f"📂 Fichier: {dataset_path}")
    
    # ✅ Charger avec allow_pickle=True pour supporter dtype=object
    data = np.load(dataset_path, allow_pickle=True)
    
    # Info générale
    print(f"\n📊 CONTENU:")
    keys = list(data.keys())
    print(f"   Nombre de clés: {len(keys)}")
    print(f"   Clés: {keys[:10]}{'...' if len(keys) > 10 else ''}")
    
    # ==========================================================
    # RÉCUPÉRER LES TRAJECTOIRES JOINTS (NOUVEAU + ANCIEN FORMAT)
    # ==========================================================
    trajectories = []
    ee_trajectories = []
    ee_present = "ee_ref_pos" in data

    if "paths" in data:
        # Nouveau format: array d'objets, chaque entrée = trajectoire (N, 6)
        paths = data["paths"]
        print("\n📦 Format détecté: 'paths' (dtype=object)")
        
        if ee_present:
            print("📦 Positions cartésiennes détectées: 'ee_ref_pos' (dtype=object)")
            ee_raw = data["ee_ref_pos"]
            if len(ee_raw) != len(paths):
                print(f"   ⚠️ 'paths' et 'ee_ref_pos' n'ont pas la même taille: {len(paths)} vs {len(ee_raw)}")
        else:
            ee_raw = None
        
        for i in range(len(paths)):
            traj = paths[i]
            if traj is None:
                continue
            traj = np.array(traj)
            trajectories.append(traj)

            if ee_present and ee_raw is not None and i < len(ee_raw):
                ee_traj = ee_raw[i]
                if ee_traj is not None:
                    ee_traj = np.array(ee_traj)
                    ee_trajectories.append(ee_traj)
    else:
        # Ancien format: traj_000, traj_001, ...
        print("\n📦 Format détecté: 'traj_XXX'")
        for key in sorted(keys):
            if key.startswith("traj_"):
                traj = np.array(data[key])
                trajectories.append(traj)
        # Ancien format: pas de ee_pos attendu
        ee_present = False

    if len(trajectories) == 0:
        print("\n❌ Aucune trajectoire trouvée!")
        return False
    
    print(f"\n✅ {len(trajectories)} trajectoires trouvées")
    
    # ===========================
    # STATISTIQUES SUR LES JOINTS
    # ===========================
    lengths = [len(t) for t in trajectories]
    print(f"\n📏 LONGUEURS (joints):")
    print(f"   Min:     {min(lengths)} points")
    print(f"   Max:     {max(lengths)} points")
    print(f"   Moyenne: {np.mean(lengths):.1f} points")
    print(f"   Médiane: {np.median(lengths):.0f} points")
    
    # Vérifier dimensions des joints
    shapes = [t.shape for t in trajectories]
    print(f"\n📐 DIMENSIONS JOINTS:")
    print(f"   Shape attendu: (N, 6) pour 6 joints")
    
    bad_shapes = [i for i, s in enumerate(shapes) if len(s) != 2 or s[1] != 6]
    if bad_shapes:
        print(f"   ❌ {len(bad_shapes)} trajectoires avec mauvaise shape:")
        for i in bad_shapes[:5]:
            print(f"      traj_{i:03d}: {shapes[i]}")
    else:
        print(f"   ✅ Toutes les trajectoires ont shape (N, 6)")
    
    # Vérifier valeurs des joints
    print(f"\n🎯 VALEURS DES JOINTS:")
    all_joints = np.vstack(trajectories)
    
    for joint_idx, joint_name in enumerate(
        ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]
    ):
        values = all_joints[:, joint_idx]
        print(f"   Joint {joint_idx} ({joint_name}):")
        print(f"      Min: {np.min(values):.3f} rad ({np.degrees(np.min(values)):.1f}°)")
        print(f"      Max: {np.max(values):.3f} rad ({np.degrees(np.max(values)):.1f}°)")
        print(f"      Std: {np.std(values):.3f} rad ({np.degrees(np.std(values)):.1f}°)")
    
    # Vérifier NaN/Inf joints
    print(f"\n🔎 VÉRIFICATION QUALITÉ (joints):")
    has_nan = np.any(np.isnan(all_joints))
    has_inf = np.any(np.isinf(all_joints))
    
    if has_nan:
        print(f"   ❌ Contient des NaN (joints)!")
    else:
        print(f"   ✅ Pas de NaN (joints)")
    
    if has_inf:
        print(f"   ❌ Contient des Inf (joints)!")
    else:
        print(f"   ✅ Pas de Inf (joints)")
    
    # Limites UR10 (larges)
    ur10_limits = [(-6.28, 6.28)] * 6
    out_of_bounds = []
    for joint_idx in range(6):
        values = all_joints[:, joint_idx]
        min_val, max_val = ur10_limits[joint_idx]
        if np.any(values < min_val) or np.any(values > max_val):
            out_of_bounds.append(joint_idx)
    
    if out_of_bounds:
        print(f"   ⚠️  Joints hors limites: {out_of_bounds}")
    else:
        print(f"   ✅ Tous les joints dans les limites")
    
    # ===========================
    # CHECK DES POSITIONS x, y, z
    # ===========================
    ee_has_nan = False
    ee_has_inf = False
    ee_bad_shapes = []
    ee_len_mismatch = []
    ee_out_of_ws = False

    if ee_present and len(ee_trajectories) > 0:
        print(f"\n📐 DIMENSIONS POSITIONS (ee_pos):")
        ee_shapes = [t.shape for t in ee_trajectories]
        print(f"   Shape attendu: (N, 3) pour (x, y, z)")
        
        for i, s in enumerate(ee_shapes):
            if len(s) != 2 or s[1] != 3:
                ee_bad_shapes.append(i)
        
        if ee_bad_shapes:
            print(f"   ❌ {len(ee_bad_shapes)} trajectoires ee_pos avec mauvaise shape:")
            for i in ee_bad_shapes[:5]:
                print(f"      ee_pos[{i}]: {ee_shapes[i]}")
        else:
            print(f"   ✅ Toutes les trajectoires ee_pos ont shape (N, 3)")
        
        # Vérifier que chaque ee_pos a la même longueur que la trajectoire joints correspondante
        print(f"\n📏 COHÉRENCE LONGUEUR joints / ee_pos:")
        min_pair = min(len(trajectories), len(ee_trajectories))
        for i in range(min_pair):
            if len(trajectories[i]) != len(ee_trajectories[i]):
                ee_len_mismatch.append(i)
        
        if ee_len_mismatch:
            print(f"   ❌ {len(ee_len_mismatch)} paires avec longueurs différentes:")
            for i in ee_len_mismatch[:5]:
                print(f"      traj[{i}] len={len(trajectories[i])}, ee_pos[{i}] len={len(ee_trajectories[i])}")
        else:
            print("   ✅ Longueur identique joints / ee_pos pour toutes les paires")
        
        # Stats sur x, y, z
        print(f"\n🌍 VALEURS DES POSITIONS (ee_pos):")
        all_pos = np.vstack(ee_trajectories)
        
        for dim, name in enumerate(["x", "y", "z"]):
            values = all_pos[:, dim]
            print(f"   {name}:")
            print(f"      Min: {np.min(values):.4f} m")
            print(f"      Max: {np.max(values):.4f} m")
            print(f"      Std: {np.std(values):.4f} m")
        
        # NaN / Inf positions
        print(f"\n🔎 VÉRIFICATION QUALITÉ (ee_pos):")
        ee_has_nan = np.any(np.isnan(all_pos))
        ee_has_inf = np.any(np.isinf(all_pos))
        
        if ee_has_nan:
            print(f"   ❌ Contient des NaN (ee_pos)!")
        else:
            print(f"   ✅ Pas de NaN (ee_pos)")
        
        if ee_has_inf:
            print(f"   ❌ Contient des Inf (ee_pos)!")
        else:
            print(f"   ✅ Pas de Inf (ee_pos)")
        
        # Vérifier par rapport au workspace si présent
        if "workspace" in data:
            ws = data["workspace"]
            print(f"\n📦 WORKSPACE:")
            print(f"   X: [{ws[0]:.3f}, {ws[1]:.3f}] m")
            print(f"   Y: [{ws[2]:.3f}, {ws[3]:.3f}] m")
            print(f"   Z: {ws[4]:.3f} m (fixe)")
            
            x_vals = all_pos[:, 0]
            y_vals = all_pos[:, 1]
            z_vals = all_pos[:, 2]
            
            x_ok = np.all((x_vals >= ws[0]) & (x_vals <= ws[1]))
            y_ok = np.all((y_vals >= ws[2]) & (y_vals <= ws[3]))
            # Z: on autorise un petit epsilon autour de la valeur fixe
            z_ok = np.all(np.isclose(z_vals, ws[4], atol=1e-3))
            
            if not x_ok or not y_ok or not z_ok:
                ee_out_of_ws = True
                print("   ⚠️ Certaines positions sont en dehors du workspace déclaré:")
                if not x_ok:
                    print("      → X hors bornes")
                if not y_ok:
                    print("      → Y hors bornes")
                if not z_ok:
                    print("      → Z différent de la valeur fixe (au-delà de 1 mm)")
            else:
                print("   ✅ Toutes les positions ee_pos sont dans le workspace")
        else:
            print("\n📦 WORKSPACE: non présent, pas de check X/Y/Z vs workspace")
    else:
        if ee_present:
            print("\n⚠️ Clé 'ee_pos' présente mais aucune trajectoire valide n'a été lue.")
        else:
            print("\nℹ️ Aucune clé 'ee_pos' détectée (pas de positions x,y,z dans ce dataset).")
    
    # Exemples joints
    print(f"\n📝 EXEMPLES (premières valeurs de 3 trajectoires - joints):")
    for i in range(min(3, len(trajectories))):
        traj = trajectories[i]
        print(f"\n   Trajectoire {i} ({len(traj)} points):")
        print(f"      Point 0: {traj[0]}")
        if len(traj) > 1:
            print(f"      Point 1: {traj[1]}")
        print(f"      Point -1: {traj[-1]}")
    
    # Exemples ee_pos
    if ee_present and len(ee_trajectories) > 0:
        print(f"\n📝 EXEMPLES (premières valeurs de 3 trajectoires - ee_pos):")
        for i in range(min(3, len(ee_trajectories))):
            traj = ee_trajectories[i]
            print(f"\n   ee_pos {i} ({len(traj)} points):")
            print(f"      Point 0: {traj[0]}  # [x, y, z]")
            if len(traj) > 1:
                print(f"      Point 1: {traj[1]}")
            print(f"      Point -1: {traj[-1]}")
    
    print("\n" + "="*70)
    is_good_joints = (not has_nan and not has_inf and len(bad_shapes) == 0 and len(trajectories) > 0)
    is_good_ee = True
    if ee_present and len(ee_trajectories) > 0:
        is_good_ee = (
            not ee_has_nan and
            not ee_has_inf and
            len(ee_bad_shapes) == 0 and
            len(ee_len_mismatch) == 0 and
            not ee_out_of_ws
        )
    
    is_good = is_good_joints and is_good_ee
    
    if is_good:
        print("✅ DATASET VALIDE - Prêt pour Isaac Lab")
    else:
        print("❌ DATASET CONTIENT DES ERREURS")
    print("="*70 + "\n")
    return is_good


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vérifier un dataset de trajectoires")
    parser.add_argument("dataset", type=str, help="Chemin vers dataset.npz")
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"❌ Fichier non trouvé: {dataset_path}")
        exit(1)
    
    check_dataset(dataset_path)
