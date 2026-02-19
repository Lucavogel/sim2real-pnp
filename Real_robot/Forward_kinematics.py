import numpy as np
import matplotlib.pyplot as plt
# Angles initiaux
theta1 = theta2 = theta3 = theta4 = theta5 = theta6 = 0.0
import time

a2 = -0.612
a3 = -0.5723
r1 = 0.1273
r2 = 0.163941
r5 = 0.1157
r7 = 0.0922  # utile si 7e paramètre
t = 0
def MatrixTransformation(uj):
    alpha, a, theta, r = uj  # Correction: d et theta inversés
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
    xx, yx, zx = R[0,0], R[0,1], R[0,2]
    xy, yy, zy = R[1,0], R[1,1], R[1,2]
    xz, yz, zz = R[2,0], R[2,1], R[2,2]

    CTheta = 0.5 * (xx + yy + zz - 1)
    STheta = 0.5 * np.sqrt((yz - zy)**2 + (zx - xz)**2 + (xy - yx)**2)

    theta = np.arctan2(STheta, CTheta)
    #TODO: quand on ne change pas d'orientation (theta proche de 0 ou pi) on devrait tourner autour de Z6
    if abs(np.sin(theta)) < 1e-9:
        u = np.array([1,0,0])
    else:
        u = np.array([
            (yz - zy),
            (zx - xz),
            (xy - yx)
        ]) / (2*np.sin(theta))

    return u, theta

def compute_jacobian(joint_pos, T06):
    q1, q2, q3, q4, q5, q6 = joint_pos
    R = T06[:3,:3]
    xx, yx, zx = R[0,0], R[0,1], R[0,2]
    xy, yy, zy = R[1,0], R[1,1], R[1,2]
    xz, yz, zz = R[2,0], R[2,1], R[2,2]
    # Correction: DH params (alpha, a, d, theta)
    T01 = np.array([0,        0,    q1, r1])
    T12 = np.array([np.pi/2, 0,    q2,  r2])
    T23 = np.array([0,      a2,    q3,  0])
    T34 = np.array([0,      a3,    q4,  0])
    T45 = np.array([np.pi/2, 0,   q5,  r5])
    T56 = np.array([-np.pi/2, 0,   q6,  0])

    T01 = MatrixTransformation(T01)
    T12 = MatrixTransformation(T12)
    T23 = MatrixTransformation(T23)
    T34 = MatrixTransformation(T34)
    T45 = MatrixTransformation(T45)
    T56 = MatrixTransformation(T56)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    # Calcul des positions et axes de rotation
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
    print("Jg =\n", np.round(Jg,6))
    J = T_tool @ Jg

    return J

def rodrigues_matrix(u, theta):
    """
    Calcule la matrice de rotation selon la formule de Rodrigues
    u: axe de rotation (vecteur unitaire)
    theta: angle de rotation
    """
    ux, uy, uz = u
    c = np.cos(theta)
    s = np.sin(theta)
    
    return np.array([
        [ux**2*(1-c)+c,    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
        [ux*uy*(1-c)+uz*s, uy**2*(1-c)+c,    uy*uz*(1-c)-ux*s],
        [ux*uz*(1-c)-uy*s, uy*uz*(1-c)+ux*s, uz**2*(1-c)+c   ]
    ])




def polynomial_trajectory(X_initial, X_end,X_actuel, u, angle,t0 ,R_initial, dt, tf=10 ):
    """
    Génère une trajectoire polynomiale de degré 5
    """
    # Time vector avec pas de temps fixe


    # Calcul du polynôme r(t)
    a3 = 10.0/((dt/tf)**3)
    a4 = -15.0/((dt/tf)**4)
    a5 = 6.0/((dt/tf)**5)

    r = a3 + a4 + a5
    r_dot = (30*((dt/tf)**2) - 60*((dt/tf)**3) + 30*((dt/tf)**4)) * (1/tf)

    #TODO: mettre vraie valeur des gains
    k_p = np.eye(3)
    k_o = np.eye(3)
    
    X_desirer = X_initial + r * (X_end - X_initial)  
    
    X_erreur = X_actuel - X_desirer
 
    X_Point_erreur = X_erreur @ k_p 
    # Dérivées de r(t)
    
    X_Point = r_dot * (X_end - X_initial)

    X_Point_erreur = X_Point_erreur + X_Point
    
    Rot = orientation(u,r,angle)
    R_desired = R_initial @ Rot

    print(R_desired)

    omega_d = R_initial @ (r_dot*angle*u)

    return X_desirer,X_Point,X_erreur,X_Point_erreur,R_desired,omega_d

def orientation(u, r, theta):
    """
    Calcule la matrice de rotation selon la formule donnée
    Args:
        u: axe de rotation (vecteur unitaire)
        r: paramètre de la trajectoire [0,1]
        theta: angle total de rotation
    """
    ux, uy, uz = u
    c = np.cos(r*theta)  # Utilisation de r*theta au lieu de theta
    s = np.sin(r*theta)  # Utilisation de r*theta au lieu de theta
    
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
def compute_error(X_current, X_desired, R_current, R_desired,X_erreur,X_Point_erreur):
    """
    Calcule l'erreur de position et d'orientation
    """


    se,ne,ae = R_current[:,0], R_current[:,1], R_current[:,2]
    sd,nd,ad = R_desired[:,0], R_desired[:,1], R_desired[:,2]

    eo = 0.5 * (np.cross(ne, nd) + np.cross(se, sd) + np.cross(ae, ad))
    print("eo =", eo)
    orientation_error = eo

    # Création des matrices antisymétriques
    S_current = [skew(R_current[:,i]) for i in range(3)]
    S_desired = [skew(R_desired[:,i]) for i in range(3)]



    L = -0.5 * (S_desired[1] @ S_current[1] + S_desired[0] @ S_current[0] + np.cross(S_desired[2],S_current[2]))


    return orientation_error,L

def MCI(eo,omega_d,J,X_Point_erreur,L,joint_pos,dt):

    K0 = np.eye(3)
    print("L =", L)
    L_inv = np.linalg.pinv(L)
    X_dot_angular = L_inv @ (L.T @ omega_d + K0 @ eo)
    print("omega_d =", omega_d)
    print("X_Point_erreur =", X_Point_erreur.size)
    print("X_dot_angular =", X_dot_angular.size)
    # Concaténation verticale
    X_Point_erreur = X_Point_erreur.reshape(3,1)  # si c'est 1D
    X_dot_angular = X_dot_angular.reshape(3,1)    # idem
    X_dot = np.array([X_Point_erreur, X_dot_angular])
    # Concaténation verticale
    X_dot = np.vstack([X_Point_erreur, X_dot_angular]) 
  

    print("X_dot =", X_dot)
    q_dot = np.linalg.pinv(J) @ X_dot
    print("q_dot =", q_dot)

    q_actuel = np.array(joint_pos)  # convertir en ndarray
    q = q_actuel.reshape(6,1) + q_dot.reshape(6,1) * dt

    q = q.flatten()  # Convertir en tableau 1D si nécessaire

    return q, q_dot

def kinematics(joint_pos,dt):
    


    print("t =", dt)

    q1, q2, q3, q4, q5, q6 = joint_pos

    # Correction: DH params (alpha, a, d, theta)
    T01 = np.array([0,        0,    q1, r1])
    T12 = np.array([np.pi/2, 0,    q2,  r2])
    T23 = np.array([0,      a2,    q3,  0])
    T34 = np.array([0,      a3,    q4,  0])
    T45 = np.array([np.pi/2, 0,   q5,  r5])
    T56 = np.array([-np.pi/2, 0,   q6,  0])

    #TODO: T06 initial avec la lecture des moteurs

    T06_initial = MatrixTransformation(np.array([0,        0,    0, r1])) @ MatrixTransformation(np.array([np.pi/2, 0,    0,  r2])) @ MatrixTransformation(np.array([0,      a2,    0,  0])) @ MatrixTransformation(np.array([0,      a3,    0,  0])) @ MatrixTransformation(np.array([np.pi/2, 0,   0,  r5])) @ MatrixTransformation(np.array([-np.pi/2, 0,   0,  0]))
    X_initial = T06_initial[:3,3]
    R_initial = T06_initial[:3,:3]
    print("T06 =\n", np.round(T06_initial,6))

    T06 = MatrixTransformation(T01) @ MatrixTransformation(T12) @ MatrixTransformation(T23) @ MatrixTransformation(T34) @ MatrixTransformation(T45) @ MatrixTransformation(T56)
    X_actuel = T06[:3,3]
    R_actuel = T06[:3,:3]
    
    u, angle = rotmat_to_axis_angle(R_initial)
    print("Axis =", u)
    print("Angle (rad) =", angle)

    J = compute_jacobian(joint_pos,T06)
    print("Jacobian J =\n", np.round(J,6))

    X_final = np.array([0.5, 0.0, 0.5])  # Position finale désirée

    X_desirer,X_Point,X_erreur,X_Point_erreur,R_desired,omega_d = polynomial_trajectory(X_initial, X_final,X_actuel, u, angle,t0 ,R_initial, dt)
    eo,L = compute_error(X_actuel, X_desirer, R_actuel, R_desired,X_erreur,X_Point_erreur)
    
    joint_pos, q_dot = MCI(eo,omega_d,J,X_Point_erreur,L,joint_pos,dt)
    print("q_actuel2 =", joint_pos)

    return joint_pos, q_dot


if __name__ == "__main__":
    a2 = -0.612
    a3 = -0.5723
    r1 = 0.1273
    r2 = 0.163941
    r5 = 0.1157
    r7 = 0.0922

    q = np.radians([5.08, -94.09, 90.75, 3.35, -184.0, 0.0])
    q1, q2, q3, q4, q5, q6 = q



    # Correction: DH params (alpha, a, d, theta)
    T01 = np.array([0,        0,    q1, r1])
    T12 = np.array([np.pi/2, 0,    q2 - np.pi/2,  r2])
    T23 = np.array([0,      a2,    q3,  0])
    T34 = np.array([0,      a3,    q4 - np.pi/2,  0])
    T45 = np.array([np.pi/2, 0,   q5,  r5])
    T56 = np.array([-np.pi/2, 0,   q6,  0])

    T01 = MatrixTransformation(T01)
    T12 = MatrixTransformation(T12)
    T23 = MatrixTransformation(T23)
    T34 = MatrixTransformation(T34)
    T45 = MatrixTransformation(T45)
    T56 = MatrixTransformation(T56)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34
    T05 = T04 @ T45
    T06 = T05 @ T56

    # Calcul des positions et axes de rotation
    transforms = [T01, T02, T03, T04, T05, T06]
    
    # Extraction des origines des repères
    origins = [T[:3,3] for T in transforms]
    P01, P02, P03, P04, P05, P06 = origins
    
    # Calcul des vecteurs position relative de l'effecteur
    end_effector = T06[:3,3]


    X_actuel = T06[:3, 3]
    print(f"x = {end_effector[0]:.6f}\ny = {end_effector[1]:.6f}\nz = {end_effector[2]:.6f}")
