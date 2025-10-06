import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sim  # CoppeliaSim remote API
import math

# Paramètres DH du robot UR10 (selon votre configuration)
a2 = -0.612
a3 = -0.5723
r1 = 0.1273
r2 = 0.163941
r5 = 0.1157
r7 = 0.0922

def MatrixTransformation(uj):
    """Matrice de transformation DH modifiée"""
    alpha, a, theta, r = uj
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)

    T = np.array([
        [ct, -st,  0, a],
        [ca*st,  ca*ct, -sa, -r*sa],
        [sa*st,   sa*ct,     ca,    r*ca],
        [0.0, 0.0,    0.0,  1.0]
    ], dtype=float)
    return T

def rotmat_to_axis_angle(R):
    """Conversion matrice de rotation vers axe-angle"""
    xx, yx, zx = R[0,0], R[0,1], R[0,2]
    xy, yy, zy = R[1,0], R[1,1], R[1,2]
    xz, yz, zz = R[2,0], R[2,1], R[2,2]

    CTheta = 0.5 * (xx + yy + zz - 1)
    STheta = 0.5 * np.sqrt((yz - zy)**2 + (zx - xz)**2 + (xy - yx)**2)

    theta = np.arctan2(STheta, CTheta)
    
    if abs(np.sin(theta)) < 1e-9:
        u = np.array([1,0,0])
    else:
        u = np.array([
            (yz - zy),
            (zx - xz),
            (xy - yx)
        ]) / (2*np.sin(theta))

    return u, theta

def compute_rotation_vector_from_matrices(R_current, R_desired):
    """
    Calcule le vecteur de rotation Δr = [Δrx, Δry, Δrz] entre deux matrices de rotation
    Ce vecteur contient l'axe (direction) et l'angle (norme) de rotation
    """
    # Matrice de rotation relative
    R_relative = R_current.T @ R_desired
    
    # Extraction de l'axe et angle
    u, theta = rotmat_to_axis_angle(R_relative)
    
    # Vecteur de rotation : Δr = θ * u
    delta_r = theta * u
    
    return delta_r

def create_orientation_matrix(orientation_type, angle_deg=0, axis=None):
    """
    Crée une matrice d'orientation 3x3 facilement.
    
    Args:
        orientation_type (str): Type d'orientation
            - "identity": matrice identité (pas de rotation)
            - "rotate_x": rotation autour de l'axe X
            - "rotate_y": rotation autour de l'axe Y  
            - "rotate_z": rotation autour de l'axe Z
            - "custom": rotation autour d'un axe personnalisé
        angle_deg (float): Angle de rotation en degrés
        axis (list/array): Axe de rotation [x,y,z] pour "custom"
    
    Returns:
        np.array: Matrice de rotation 3x3
        
    Examples:
        R = create_orientation_matrix("rotate_z", 90)      # 90° autour Z
        R = create_orientation_matrix("custom", 45, [1,1,0])  # 45° autour [1,1,0]
    """
    if orientation_type == "identity":
        return np.eye(3)
    
    angle_rad = np.radians(angle_deg)
    
    if orientation_type == "rotate_x":
        axis = [1, 0, 0]
    elif orientation_type == "rotate_y":
        axis = [0, 1, 0]
    elif orientation_type == "rotate_z":
        axis = [0, 0, 1]
    elif orientation_type == "custom":
        if axis is None:
            raise ValueError("L'axe doit être spécifié pour une rotation personnalisée")
        axis = np.array(axis) / np.linalg.norm(axis)  # Normalisation
    else:
        raise ValueError(f"Type d'orientation non reconnu: {orientation_type}")
    
    return rodrigues_matrix(axis, angle_rad)

def afficher_guide_coordonnees(X_current, R_current):
    """
    Affiche un guide pratique pour choisir les coordonnées finales.
    """
    print("\n" + "="*60)
    print("🎯 GUIDE POUR SPÉCIFIER LES COORDONNÉES FINALES")
    print("="*60)
    
    print(f"\n📍 POSITION ACTUELLE DU ROBOT :")
    print(f"   X = [{X_current[0]:.3f}, {X_current[1]:.3f}, {X_current[2]:.3f}] m")
    
    print(f"\n🔧 MÉTHODES DISPONIBLES :")
    print(f"   1️⃣  COORDONNÉES DIRECTES (simple)")
    print(f"       X_final = np.array([-0.4, -0.3, 0.4])")
    print(f"       R_final = create_orientation_matrix('rotate_z', 90)")
    
    print(f"\n   2️⃣  DÉPLACEMENT RELATIF (pratique)")
    print(f"       X_final = X_initial + np.array([0.1, 0, 0])  # +10cm en X")
    print(f"       R_final = R_initial @ create_orientation_matrix('rotate_x', 45)")
    
    print(f"\n   3️⃣  CONFIGURATION ARTICULAIRE (précise)")
    print(f"       Q_final = joint_pos + np.radians([10, 0, 0, 0, 0, 0])  # +10° joint 1")
    
    print(f"\n🎨 ORIENTATIONS PRÉDÉFINIES :")
    print(f"   • create_orientation_matrix('identity')        → Pas de rotation")
    print(f"   • create_orientation_matrix('rotate_x', 90)    → 90° autour X")
    print(f"   • create_orientation_matrix('rotate_y', -45)   → -45° autour Y")  
    print(f"   • create_orientation_matrix('rotate_z', 180)   → 180° autour Z")
    print(f"   • create_orientation_matrix('custom', 30, [1,1,0]) → 30° autour [1,1,0]")
    
    print(f"\n⚠️  ZONE DE TRAVAIL SÉCURISÉE :")
    print(f"   X ∈ [-0.6, +0.6] m")
    print(f"   Y ∈ [-0.6, +0.6] m") 
    print(f"   Z ∈ [+0.2, +0.8] m")
    
    print(f"\n✅ POSITIONS RECOMMANDÉES POUR TESTS :")
    print(f"   • Position proche : X_final = X_initial + [0.1, 0.1, 0.05]")
    print(f"   • Position sûre   : X_final = [-0.4, -0.3, 0.4]")
    print(f"   • Position haute  : X_final = [-0.3, -0.2, 0.6]")
    
    print("="*60)

def compute_jacobian(joint_pos, T06):
    q1, q2, q3, q4, q5, q6 = joint_pos
    R = T06[:3,:3]
    xx, yx, zx = R[0,0], R[0,1], R[0,2]
    xy, yy, zy = R[1,0], R[1,1], R[1,2]
    xz, yz, zz = R[2,0], R[2,1], R[2,2]
    # Correction: DH params (alpha, a, d, theta)
    T01 = np.array([0,        0,    q1, r1])
    T12 = np.array([np.pi/2 , 0,    q2 - np.pi/2 ,  r2])
    T23 = np.array([0,      a2,    q3,  0])
    T34 = np.array([ 0,      a3,    q4 - np.pi/2 ,  0])
    T45 = np.array([np.pi/2, 0,   q5,  r5])
    T56 = np.array([-np.pi/2, 0,   q6,  0])
    T = np.array([T01, T12, T23, T34, T45, T56])    
    # Calcul des matrices de transformation individuelles
    T01 = MatrixTransformation(T01)
    T12 = MatrixTransformation(T12)
    T23 = MatrixTransformation(T23)
    T34 = MatrixTransformation(T34)
    T45 = MatrixTransformation(T45)
    T56 = MatrixTransformation(T56)

    # Calcul des transformations cumulées
    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    # Organisation des transformations pour le calcul de la Jacobienne
    transforms = [T01, T02, T03, T04, T05, T06]
    
    # Extraction des origines des repères
    origins = [T[:3,3] for T in transforms]
    P01, P02, P03, P04, P05, P06 = origins
    
    # Calcul des vecteurs position relative de l'effecteur
    end_effector = T06[:3,3]
    rel_positions = [end_effector - origin for origin in origins]
    P014, P024, P034, P044, P054, P064 = rel_positions
    
    # Extraction des axes Z (axes de rotation)
    z_axes = [T[:3,2] for T in transforms]
    Z01, Z02, Z03, Z04, Z05, Z06 = z_axes
    
    # Calcul des composantes de la Jacobienne
    # Partie linéaire (vitesse)
    Jv = [np.cross(z, p) for z, p in zip(z_axes, rel_positions)]
    
    # Partie angulaire (rotation)
    Jw = z_axes
    
    # Assemblage de la matrice Jacobienne géométrique
    Jg = np.zeros((6,6))
    Jg[:3] = np.column_stack(Jv)  
    Jg[3:] = np.column_stack(Jw)  

   
    r7 = 0.0922  # Décalage sur l'axe Z de l'effecteur
    an = 0
    # Matrice antisymétrique pour le décalage r7 sur Z
    D = np.array([
        [0,   an*xz+r7*zz,  -an*xy-r7*zy],
        [-an*xz-r7*zz,    0,  an*xx + r7*zx],
        [an*xy+r7*zy,     -an*xx-r7*zx,  0]
    ])
    
    # Construction de la matrice de transformation complète
    I = np.eye(3)  # Matrice identité 3x3
    O = np.zeros((3,3))  # Matrice nulle 3x3
    
    # Assemblage de la matrice de transformation [I D; 0 C]
    T_tool = np.vstack([
        np.hstack([I, D]),
        np.hstack([O, I])
    ])
    
    # Application de la transformation à la Jacobienne géométrique
    J = T_tool @ Jg

    return J

def rodrigues_matrix(u, theta):
    """Matrice de rotation selon Rodrigues"""
    ux, uy, uz = u
    c = np.cos(theta)
    s = np.sin(theta)
    
    return np.array([
        [ux**2*(1-c)+c,    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [ux*uy*(1-c)+uz*s, uy**2*(1-c)+c,    uy*uz*(1-c)-ux*s],
        [ux*uz*(1-c)-uy*s, uy*uz*(1-c)+ux*s, uz**2*(1-c)+c   ]
    ])

def polynomial_trajectory(X_initial, X_end, X_actuel, u, angle, t_current, R_initial, dt, tf=10, R_final=None):
    """Génère une trajectoire polynomiale de degré 5 avec orientation finale"""
    # Normalisation du temps entre 0 et 1
    base_gain = 1.0 / dt
    s = t_current / tf
    
    # Polynôme de degré 5: r(s) = 10s³ - 15s⁴ + 6s⁵
    r = 10*s**3 - 15*s**4 + 6*s**5
    r_dot = (30*s**2 - 60*s**3 + 30*s**4) / tf
    
    # Position désirée
    X_desirer = X_initial + r * (X_end - X_initial)
    
    # Erreur de position (CORRECTION: désirée - actuelle)
    X_erreur = X_desirer - X_actuel

    # Gains proportionnels basés sur 1/dt
    Kp = base_gain * np.eye(3)  # Gain Kp = 1/dt pour trajectoire
    
    # Vitesse désirée + correction proportionnelle
    X_Point_desirer = r_dot * (X_end - X_initial)
    X_Point_erreur = X_Point_desirer + Kp @ X_erreur
    print("Norme de u :", np.linalg.norm(u))
    Rot = orientation(u, r, angle)
    R_desired = R_initial @ Rot  # R_initial * rot(u, r*θ) pour interpolation
        
    # Vitesse angulaire désirée pour transition vers R_final
    omega_d = R_initial @ (r_dot * angle * u)
  
    
    return X_desirer, X_Point_erreur, X_erreur, R_desired, omega_d

def polynomial_trajectory_simplified(X_initial, X_end, X_actuel, t_current, R_initial, R_final, dt, tf=10):
    """
    Trajectoire simplifiée utilisant directement le vecteur de rotation
    """
    # Normalisation du temps
    base_gain = 1.0 / dt
    s = t_current / tf
    
    # Polynôme de degré 5
    r = 10*s**3 - 15*s**4 + 6*s**5
    r_dot = (30*s**2 - 60*s**3 + 30*s**4) / tf
    
    # Trajectoire de position
    X_desirer = X_initial + r * (X_end - X_initial)
    X_erreur = X_desirer - X_actuel
    
    # Gains
    Kp = base_gain * np.eye(3)
    X_Point_desirer = r_dot * (X_end - X_initial)
    X_Point_erreur = X_Point_desirer + Kp @ X_erreur
    
    # 🎯 NOUVEAU : Vecteur de rotation direct
    delta_r_total = compute_rotation_vector_from_matrices(R_initial, R_final)
    
    # Interpolation du vecteur de rotation
    delta_r_desired = r * delta_r_total  # Interpolation linéaire
    delta_r_dot = r_dot * delta_r_total  # Vitesse de rotation
    
    # Matrice de rotation désirée (via Rodriguez)
    if np.linalg.norm(delta_r_desired) > 1e-6:
        angle_desired = np.linalg.norm(delta_r_desired)
        axis_desired = delta_r_desired / angle_desired
        R_desired = R_initial @ rodrigues_matrix(axis_desired, angle_desired)
    else:
        R_desired = R_initial.copy()
    
    return X_desirer, X_Point_erreur, X_erreur, R_desired, delta_r_dot

def orientation(u, r, theta):
    """Calcule la matrice de rotation selon la formule donnée"""
    ux, uy, uz = u
    c = np.cos(r*theta)
    s = np.sin(r*theta)
    
    return np.array([
        [ux**2*(1-c)+c,    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [ux*uy*(1-c)+uz*s, uy**2*(1-c)+c,    uy*uz*(1-c)-ux*s],
        [ux*uz*(1-c)-uy*s, uy*uz*(1-c)+ux*s, uz**2*(1-c)+c   ]
    ])

def skew(v):
    """Retourne la matrice antisymétrique 3x3 d'un vecteur 3x1"""
    x, y, z = v
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]])

def compute_error(X_current, X_desired, R_current, R_desired):
    """Calcule l'erreur de position et d'orientation"""
    # Erreur de position
    position_error = X_desired - X_current
    
    # Erreur d'orientation selon la formule du TP
    se, ne, ae = R_current[:, 0], R_current[:, 1], R_current[:, 2]
    sd, nd, ad = R_desired[:, 0], R_desired[:, 1], R_desired[:, 2]
    
    eo = 0.5 * (np.cross(ne, nd) + np.cross(se, sd) + np.cross(ae, ad))
    
    # Calcul de la matrice L selon la formule du TP dans UR-10_save
    # Création des matrices antisymétriques
    S_current = [skew(R_current[:,i]) for i in range(3)]
    S_desired = [skew(R_desired[:,i]) for i in range(3)]

    L = -0.5 * (S_desired[1] @ S_current[1] + S_desired[0] @ S_current[0] + S_desired[2] @ S_current[2])

    return position_error, eo, L

def compute_error_simplified(X_current, X_desired, R_current, R_desired):
    """
    Calcul d'erreur simplifié avec vecteur de rotation direct
    """
    # Erreur de position (inchangée)
    position_error = X_desired - X_current
    
    # 🎯 NOUVEAU : Erreur d'orientation comme vecteur de rotation
    delta_r_error = compute_rotation_vector_from_matrices(R_current, R_desired)
    
    # Matrice L simplifiée (identité pour cette approche)
    L = np.eye(3)
    
    return position_error, delta_r_error, L

def MCI(position_error, eo, omega_d, J, X_Point_erreur, L, joint_pos, dt):
    """Loi de commande (Modèle de Commande Inverse)"""
    # Gains proportionnels basés sur 1/dt pour compensation temporelle
    base_gain = 1.0 / dt
    Kp = base_gain * np.eye(3)  # Gain Kp = 1/dt
    Ko = base_gain * np.eye(3)  # Gain Ko = 1/dt

    # Vitesse désirée en position avec correction d'erreur
    X_dot_linear = X_Point_erreur + Kp @ position_error
    
    # Vitesse angulaire désirée avec correction d'erreur - avec robustesse # Seuil de robustesse pour pseudo-inverse
    L_inv = np.linalg.pinv(L)
    X_dot_angular = L_inv @ (L.T @ omega_d + Ko @ eo)
    
    # Assemblage du vecteur vitesse 6D
    X_dot = np.vstack([X_dot_linear.reshape(-1, 1), X_dot_angular.reshape(-1, 1)])
    
    # Vérification des singularités
    cond_J = np.linalg.cond(J)
    cond_L = np.linalg.cond(L)
    
 
    
    # Calcul des vitesses articulaires avec robustesse
    J_inv = np.linalg.pinv(J)
    q_dot = J_inv @ X_dot
    q_dot = q_dot.flatten()
    

    
    # Intégration pour obtenir les nouvelles positions
    q_actuel = np.array(joint_pos)
    print(q_actuel)
    q_new = q_actuel + q_dot * dt
    
    return q_new, q_dot

def MCI_simplified(position_error, delta_r_error, delta_r_dot, J, X_Point_erreur, joint_pos, dt):
    """
    MCI simplifié utilisant directement le vecteur de rotation
    """
    # Paramètre de régularisation pour la Jacobienne
    k_damping = 0.01  # Paramètre de Levenberg-Marquardt
    
    # Gains basés sur 1/dt
    base_gain = 1.0 / dt
    Kp = base_gain * np.eye(3)
    Ko = base_gain * np.eye(3)
    
    # Vitesse linéaire désirée
    X_dot_linear = X_Point_erreur + Kp @ position_error
    
    # 🎯 NOUVEAU : Vitesse angulaire directe
    X_dot_angular = delta_r_dot + Ko @ delta_r_error
    
    # Assemblage du vecteur vitesse 6D
    X_dot = np.vstack([X_dot_linear.reshape(-1, 1), X_dot_angular.reshape(-1, 1)])
    
    # 🎯 NOUVEAU : Pseudo-inverse régularisée de Levenberg-Marquardt
    # J* = J^T(JJ^T + k²I)^(-1)
    JJT = J @ J.T
    I6 = np.eye(6)
    J_damped_inv = J.T @ np.linalg.inv(JJT + (k_damping**2) * I6)
    
    # Calcul des vitesses articulaires
    q_dot = J_damped_inv @ X_dot
    q_dot = q_dot.flatten()
    
    # Nouvelle position (pour CoppeliaSim en position)
    q_new = np.array(joint_pos) + q_dot * dt
    
    return q_new, q_dot

def kinematics(joint_pos):
    """
    Fonction principale selon le TP:
    - Q1: Calcul des matrices de transformation DH
    - Q2: Position et orientation de l'effecteur (T06) + représentation axe-angle
    - Q3: Modèle cinématique différentiel (Jacobienne)
    """
    q1, q2, q3, q4, q5, q6 = joint_pos

    # Q1: Paramètres DH (alpha, a, theta, r) selon convention modifiée
    T01_params = np.array([0,        0,    q1, r1])
    T12_params = np.array([np.pi/2,  0,    q2 - np.pi/2, r2])
    T23_params = np.array([0,       a2,    q3,  0])
    T34_params = np.array([0,       a3,    q4 - np.pi/2,  0])
    T45_params = np.array([np.pi/2,  0,    q5, r5])
    T56_params = np.array([-np.pi/2, 0,    q6,  0])

    # Calcul des matrices de transformation individuelles
    T01 = MatrixTransformation(T01_params)
    T12 = MatrixTransformation(T12_params)
    T23 = MatrixTransformation(T23_params)
    T34 = MatrixTransformation(T34_params)
    T45 = MatrixTransformation(T45_params)
    T56 = MatrixTransformation(T56_params)

    # Q2: Calcul de T06 - Position et orientation de l'effecteur
    T06 = T01 @ T12 @ T23 @ T34 @ T45 @ T56
    
    # Q3: Calcul de la Jacobienne
    J = compute_jacobian(joint_pos, T06)
    
    return T06, J



def create_comprehensive_plots(time_history, joint_history, position_history, position_desired_history, 
                              error_history, orientation_error_history, velocity_history, jacobian_cond_history, 
                              orientation_error_components=None, jacobian_det_history=None):
    """Crée des graphiques essentiels pour démontrer le fonctionnement du système"""
    
    # Figure principale - graphiques essentiels (3x3 pour plus de graphiques)
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('SYSTÈME UR10 - VALIDATION COMPLÈTE', fontsize=16, fontweight='bold')
    
    # 1. Évolution des angles des joints
    plt.subplot(3, 3, 1)
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
    for j in range(6):
        plt.plot(time_history, np.degrees(joint_history[:, j]), 
                color=colors[j], linewidth=2, label=f'Joint {j+1}')
    plt.xlabel('Temps (s)')
    plt.ylabel('Angle (°)')
    plt.title('Angles des Joints')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Position X: Actuelle vs Désirée
    plt.subplot(3, 3, 2)
    plt.plot(time_history, position_history[:, 0], 'r-', linewidth=2, label='X réel')
    plt.plot(time_history, position_desired_history[:, 0], 'r--', linewidth=1, alpha=0.7, label='X désiré')
    plt.plot(time_history, position_history[:, 1], 'g-', linewidth=2, label='Y réel')
    plt.plot(time_history, position_desired_history[:, 1], 'g--', linewidth=1, alpha=0.7, label='Y désiré')
    plt.plot(time_history, position_history[:, 2], 'b-', linewidth=2, label='Z réel')
    plt.plot(time_history, position_desired_history[:, 2], 'b--', linewidth=1, alpha=0.7, label='Z désiré')
    plt.xlabel('Temps (s)')
    plt.ylabel('Position (m)')
    plt.title('Suivi de Trajectoire')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Erreurs de position par axe
    plt.subplot(3, 3, 3)
    pos_errors = np.abs(position_desired_history - position_history) * 1000  # En mm
    plt.plot(time_history, pos_errors[:, 0], 'r-', linewidth=2, label='Erreur X')
    plt.plot(time_history, pos_errors[:, 1], 'g-', linewidth=2, label='Erreur Y')
    plt.plot(time_history, pos_errors[:, 2], 'b-', linewidth=2, label='Erreur Z')
    plt.xlabel('Temps (s)')
    plt.ylabel('Erreur position (mm)')
    plt.title('Erreurs Position par Axe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Trajectoire 3D
    ax1 = plt.subplot(2, 3, 4, projection='3d')
    ax1.plot(position_history[:, 0], position_history[:, 1], position_history[:, 2], 
             'b-', linewidth=2, label='Trajectoire réelle')
    ax1.plot(position_desired_history[:, 0], position_desired_history[:, 1], position_desired_history[:, 2], 
             'r--', linewidth=1, alpha=0.7, label='Trajectoire désirée')
    ax1.scatter(position_history[0, 0], position_history[0, 1], position_history[0, 2], 
               color='green', s=100, label='Départ')
    ax1.scatter(position_history[-1, 0], position_history[-1, 1], position_history[-1, 2], 
               color='red', s=100, label='Arrivée')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('Trajectoire 3D')
    ax1.legend()
    
    # 5. Erreurs d'orientation par axe
    plt.subplot(2, 3, 5)
    if orientation_error_components is not None:
        plt.plot(time_history, orientation_error_components[:, 0], 'r-', linewidth=2, label='Erreur ex')
        plt.plot(time_history, orientation_error_components[:, 1], 'g-', linewidth=2, label='Erreur ey')
        plt.plot(time_history, orientation_error_components[:, 2], 'b-', linewidth=2, label='Erreur ez')
    else:
        # Fallback si pas de données de composantes
        plt.plot(time_history, velocity_history, 'purple', linewidth=2, label='Vitesses')
    plt.xlabel('Temps (s)')
    plt.ylabel('Erreur orientation')
    plt.title('Erreurs Orientation par Axe')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Déterminant de la Jacobienne avec détecteur de singularités
    plt.subplot(2, 3, 6)
    if jacobian_det_history is not None:
        # Tracé du déterminant
        plt.plot(time_history, jacobian_det_history, 'b-', linewidth=2, label='det(J)')
        
        # Seuil de singularité (valeur absolue proche de 0)
        singularity_threshold = 1e-3
        plt.axhline(y=singularity_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Seuil sing. (+{singularity_threshold})')
        plt.axhline(y=-singularity_threshold, color='red', linestyle='--', linewidth=1, alpha=0.7, label=f'Seuil sing. (-{singularity_threshold})')
        
        # Zone de singularité (en rouge)
        mask_singular = np.abs(jacobian_det_history) < singularity_threshold
        if np.any(mask_singular):
            plt.fill_between(time_history, np.min(jacobian_det_history), np.max(jacobian_det_history), 
                           where=mask_singular, alpha=0.2, color='red', label='Zone singulière')
            
        # Détection et marquage des singularités
        singular_points = np.where(mask_singular)[0]
        if len(singular_points) > 0:
            plt.scatter(time_history[singular_points], jacobian_det_history[singular_points], 
                       color='red', s=50, marker='x', linewidth=3, label='Singularités détectées')
            
        # Statistiques des singularités
        singular_percentage = (np.sum(mask_singular) / len(jacobian_det_history)) * 100
        min_det = np.min(np.abs(jacobian_det_history))
        
        plt.xlabel('Temps (s)')
        plt.ylabel('Déterminant J')
        plt.title(f'Détection Singularités ({singular_percentage:.1f}% du temps)')
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        
        # Affichage du minimum
        plt.text(0.02, 0.98, f'|det(J)|_min = {min_det:.2e}', 
                transform=plt.gca().transAxes, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        plt.text(0.5, 0.5, 'Données déterminant\nnon disponibles', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Déterminant Jacobienne')
        plt.axis('off')
    
    plt.tight_layout()
    
    return fig

if __name__ == "__main__":
    # Code CoppeliaSim basique (comme ur10_etudiant.py)
    print('Programme démarré')
    sim.simxFinish(-1)  # Fermer toutes les connexions ouvertes
    clientID = sim.simxStart('127.0.0.1', 19999, True, True, 5000, 5)  # Connexion à CoppeliaSim

    h = np.array([0, 0, 0, 0, 0, 0])
    q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    if clientID != -1:
        print('Connecté au serveur API distant')

        # Récupérer les données de manière bloquante
        res, objs = sim.simxGetObjects(clientID, sim.sim_handle_all, sim.simx_opmode_blocking)
        if res == sim.simx_return_ok:
            print(f'Nombre d\'objets dans la scène: {len(objs)}')
        else:
            print(f'Erreur de fonction API: {res}')
            
        # Récupérer les données en streaming
        startTime = time.time()
        sim.simxGetIntegerParameter(clientID, sim.sim_intparam_mouse_x, sim.simx_opmode_streaming)
           
        r, h[0] = sim.simxGetObjectHandle(clientID, 'UR10_joint1', sim.simx_opmode_blocking)
        r, h[1] = sim.simxGetObjectHandle(clientID, 'UR10_joint2', sim.simx_opmode_blocking)
        r, h[2] = sim.simxGetObjectHandle(clientID, 'UR10_joint3', sim.simx_opmode_blocking)
        r, h[3] = sim.simxGetObjectHandle(clientID, 'UR10_joint4', sim.simx_opmode_blocking)
        r, h[4] = sim.simxGetObjectHandle(clientID, 'UR10_joint5', sim.simx_opmode_blocking)
        r, h[5] = sim.simxGetObjectHandle(clientID, 'UR10_joint6', sim.simx_opmode_blocking)
        
        #############################################################################################################
        #                       Programme de la commande débute ici
        #############################################################################################################

        # Changer une variable articulaire : exemple sur la première articulaire (valeur = 20°)
        #for i in range(6):
        sim.simxSetJointTargetPosition(clientID, h[2], math.radians(20), sim.simx_opmode_blocking)
        
        sim.simxSetJointTargetPosition(clientID, h[4], math.radians(20), sim.simx_opmode_blocking)
        print("Joint 1 défini à 20°")
        
        time.sleep(2)  # Attendre le mouvement
        
        # Lire une variable articulaire : exemple sur la première articulaire
        r, theta1 = sim.simxGetJointPosition(clientID, h[0], sim.simx_opmode_blocking)
        print(f"Position lue du joint 1: {theta1:.3f} rad ({np.degrees(theta1):.1f}°)")
        
        # Maintenant lancer le contrôleur avec CoppeliaSim
        print("\n=== CONTRÔLEUR AVEC COPPELIA ===")
        
        # Paramètres de simulation
        tf = 10.0
        dt_target = 0.02  # 50Hz pour CoppeliaSim
        
        # Lire les positions initiales des joints depuis CoppeliaSim
        joint_pos = np.zeros(6)
        for i in range(6):
            r, joint_pos[i] = sim.simxGetJointPosition(clientID, h[i], sim.simx_opmode_blocking)
        
        print(f"Positions initiales des joints: {np.round(np.degrees(joint_pos), 1)} °")
        
        # Position initiale de l'effecteur
        T06_init, J_init = kinematics(joint_pos)
        det_J = np.linalg.det(J_init)
        print(f"Déterminant de la Jacobienne (singularité si ~0): {det_J:.6f}")
        #TODO pas X_Initial mais 6o7
        O_T = np.array([0,0,r7,1])
        O_T = O_T.reshape(4,1)
        T_Final = T06_init @ O_T
        X_initial = T_Final[:3].flatten()

        R_initial = T06_init[:3, :3]
        
        print(f"Position initiale effecteur: {np.round(X_initial, 3)}")
        
        # 📚 Affichage du guide pratique pour choisir les coordonnées
        afficher_guide_coordonnees(X_initial, R_initial)
        
        # 🎯 CHOIX DES COORDONNÉES FINALES - Choisissez UNE méthode :
        # 
        # 💡 GUIDE PRATIQUE :
        # - Position actuelle du robot : X = [{:.3f}, {:.3f}, {:.3f}] m
        # - Pour un déplacement simple : utilisez la MÉTHODE 2
        # - Pour une orientation précise : utilisez create_orientation_matrix()
        # - Zone de travail sécurisée : X ∈ [-0.6, 0.6], Y ∈ [-0.6, 0.6], Z ∈ [0.2, 0.8]
        
        # MÉTHODE 1: Coordonnées cartésiennes directes (RECOMMANDÉE)
        X_final = np.array([-0.5, -0.4, 0.3])  # [X, Y, Z] en mètres
        R_final = create_orientation_matrix("rotate_x", 90)  # 90° autour Z
        
        # MÉTHODE 2: Position relative + orientation personnalisée
        # X_final = X_initial + np.array([0.1, 0.1, 0.05])  # Déplacement [ΔX, ΔY, ΔZ]
        # R_final = R_initial @ create_orientation_matrix("rotate_x", 45)  # +45° autour X
        
        # MÉTHODE 3: À partir d'une configuration articulaire
        # Q_final = joint_pos + np.radians([10, 10, 10, 10, 10, 10])  # Angles en degrés
        # T06_final, J_final = kinematics(Q_final)
        # X_final = T06_final[:3, 3]
        # R_final = T06_final[:3, :3]
        
        # MÉTHODE 4: Exemples d'orientations prédéfinies
        # R_final = create_orientation_matrix("identity")           # Pas de rotation
        # R_final = create_orientation_matrix("rotate_x", 90)      # 90° autour X
        # R_final = create_orientation_matrix("rotate_y", -45)     # -45° autour Y
        # R_final = create_orientation_matrix("custom", 30, [1,1,0])  # 30° autour axe [1,1,0]
        
        # MÉTHODE 5: Matrice d'orientation manuelle (pour experts)
        # R_final = np.array([[1, 0, 0],    # Vecteur X de l'effecteur
        #                    [0, -1, 0],    # Vecteur Y de l'effecteur  
        #                    [0, 0, -1]])   # Vecteur Z de l'effecteur
        
        print(f"Position désirée: {np.round(X_final, 3)}") 

        # 🎯 MÉTHODE TRADITIONNELLE : Calcul de l'axe et angle séparément
        R_relative = R_initial.T @ R_final
        u, angle = rotmat_to_axis_angle(R_relative)
        print(f"Méthode traditionnelle - Axe: {np.round(u, 3)}, Angle: {np.round(np.degrees(angle), 1)}°")
        
        # 🚀 NOUVELLE MÉTHODE : Vecteur de rotation direct
        delta_r = compute_rotation_vector_from_matrices(R_initial, R_final)
        print(f"Nouvelle méthode - Vecteur rotation: {np.round(delta_r, 3)}")
        print(f"  → Norme (angle): {np.round(np.degrees(np.linalg.norm(delta_r)), 1)}°")
        print(f"  → Direction (axe): {np.round(delta_r/np.linalg.norm(delta_r), 3)}")
        
        # Stockage pour l'affichage
        joint_history = []
        time_history = []
        position_history = []
        position_desired_history = []
        error_history = []
        orientation_error_history = []
        orientation_error_components = []  # Nouvelles erreurs d'orientation par composante
        velocity_history = []
        jacobian_cond_history = []
        jacobian_det_history = []  # Stockage du déterminant pour détection de singularités
        
        # Simulation temps réel avec CoppeliaSim
        t_start = time.time()
        print("Contrôle temps réel avec CoppeliaSim en cours...")
        
        while True:
            t_current = time.time() - t_start
            
            # Arrêt si temps final atteint
            if t_current >= tf:
                break
            
            # Lire les positions actuelles depuis CoppeliaSim
            joint_pos_actual = np.zeros(6)
            for i in range(6):
                r, joint_pos_actual[i] = sim.simxGetJointPosition(clientID, h[i], sim.simx_opmode_blocking)
            
            # Cinématique directe basée sur les positions réelles
            T06, J = kinematics(joint_pos_actual)
            # Calcul position effecteur avec offset organe terminal
            O_T = np.array([0,0,r7,1]).reshape(4,1)
            T_Final = T06 @ O_T
            X_actuel = T_Final[:3].flatten()
            R_actuel = T06[:3, :3]
            
            # 📍 MÉTHODE TRADITIONNELLE (actuellement utilisée)
            X_desirer, X_Point_erreur, X_erreur, R_desired, omega_d = polynomial_trajectory(
                X_initial, X_final, X_actuel, u, angle, t_current, R_initial, dt_target, tf, R_final)
            position_error, eo, L = compute_error(X_actuel, X_desirer, R_actuel, R_desired)
            joint_pos_new, q_dot = MCI(position_error, eo, omega_d, J, X_Point_erreur, L, joint_pos_actual, dt_target)
            
            # 🚀 NOUVELLE MÉTHODE SIMPLIFIÉE (à activer si désiré)
            # Décommentez les lignes suivantes pour utiliser l'approche simplifiée :
            """
            X_desirer, X_Point_erreur, X_erreur, R_desired, delta_r_dot = polynomial_trajectory_simplified(
                X_initial, X_final, X_actuel, t_current, R_initial, R_final, dt_target, tf)
            position_error, delta_r_error, L_simple = compute_error_simplified(X_actuel, X_desirer, R_actuel, R_desired)
            joint_pos_new, q_dot = MCI_simplified(position_error, delta_r_error, delta_r_dot, J, X_Point_erreur, joint_pos_actual, dt_target)
            """
            
            # Envoyer les nouvelles positions aux joints dans CoppeliaSim
            for i in range(6):
                sim.simxSetJointTargetPosition(clientID, h[i], joint_pos_new[i], sim.simx_opmode_streaming)
            
            # Stockage des données
            joint_history.append(joint_pos_actual.copy())
            time_history.append(t_current)
            position_history.append(X_actuel.copy())
            position_desired_history.append(X_desirer.copy())
            error_history.append(np.linalg.norm(position_error))
            orientation_error_history.append(np.linalg.norm(eo))
            orientation_error_components.append(eo.copy())  # Stockage des composantes individuelles
            velocity_history.append(np.linalg.norm(q_dot))
            jacobian_cond_history.append(np.linalg.cond(J))
            jacobian_det_history.append(np.linalg.det(J))  # Stockage du déterminant
            
            # Respect de la fréquence cible
            time.sleep(dt_target)
            
            # Affichage périodique détaillé
            if len(time_history) % 25 == 0:
                print(f"t={t_current:.2f}s")
                print(f"  Pos: {np.round(X_actuel, 3)} m")
                print(f"  Err_pos: X={position_error[0]*1000:.1f}mm, Y={position_error[1]*1000:.1f}mm, Z={position_error[2]*1000:.1f}mm")
                print(f"  Err_ori: ex={eo[0]:.4f}, ey={eo[1]:.4f}, ez={eo[2]:.4f}")
                print(f"  ||Err||: pos={np.linalg.norm(position_error)*1000:.1f}mm, ori={np.linalg.norm(eo):.4f}")
        
        # Conversion en arrays numpy
        joint_history = np.array(joint_history)
        time_history = np.array(time_history)
        position_history = np.array(position_history)
        position_desired_history = np.array(position_desired_history)
        error_history = np.array(error_history)
        orientation_error_history = np.array(orientation_error_history)
        orientation_error_components = np.array(orientation_error_components)
        velocity_history = np.array(velocity_history)
        jacobian_cond_history = np.array(jacobian_cond_history)
        jacobian_det_history = np.array(jacobian_det_history)
        
        print(f"\n=== RÉSULTATS COPPELIA ===")
        print(f"Position finale: {np.round(position_history[-1], 3)}")
        print(f"Erreur finale: {np.round(error_history[-1], 4)} m")
        print(f"Erreur max: {np.round(np.max(error_history), 4)} m")
        
        # Génération des graphiques pour CoppeliaSim
        print("\n=== Génération graphiques CoppeliaSim ===")
        fig = create_comprehensive_plots(time_history, joint_history, position_history, position_desired_history, 
                                         error_history, orientation_error_history, velocity_history, jacobian_cond_history, orientation_error_components, jacobian_det_history)
        
        # Validation finale
        print("\n=== VALIDATION COPPELIA ===")
        if error_history[-1] < 0.01:
            print("✅ SYSTÈME VALIDÉ AVEC COPPELIA")
        else:
            print("⚠️  SYSTÈME À AMÉLIORER")
            
        print(f"Score CoppeliaSim: {(1 - np.mean(error_history))*100:.1f}%")
        
        plt.show()
        
        # Remettre tous les joints à 0
        print("\n=== RÉINITIALISATION ===")
        for i in range(6):
            sim.simxSetJointTargetPosition(clientID, h[i], 0, sim.simx_opmode_blocking)
        
        print("Tous les joints remis à zéro")
        
        # Fermer la connexion
        sim.simxFinish(clientID)
        print("Connexion fermée")
            
    else:
        print('Échec de connexion au serveur API distant')
        print('Le programme nécessite CoppeliaSim pour fonctionner')
    
    print('Programme terminé')