import numpy as np
import cv2
import glob
import os

# --- CONFIGURATION ---
CHECKERBOARD = (8, 6) # coins internes (colonnes, lignes) - PAS les carrés !
SQUARE_SIZE = 0.025   # mètres (Mesurez votre feuille imprimée !)
CAMERA_ID = 3         # L'index de votre webcam

def run_calibration():
    # 1. Configuration
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = objp * SQUARE_SIZE

    objpoints = [] # points 3d dans l'espace réel
    imgpoints = [] # points 2d dans le plan image

    cap = cv2.VideoCapture(CAMERA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("📸 MODE CALIBRATION")
    print("   Appuyez sur 'c' pour capturer une frame (visez 20-30 frames variées).")
    print("   Appuyez sur 'q' pour terminer et calculer.")

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Trouver les coins pour la prévisualisation
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        # Dessiner le retour visuel
        display = frame.copy()
        if ret_corners:
            cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_corners)
            cv2.putText(display, "Coins Trouves!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(display, f"Captures: {count}", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow('Calibration', display)

        key = cv2.waitKey(1)
        if key == ord('c'):
            if ret_corners:
                objpoints.append(objp)
                
                # Raffiner les coins (La précision subpixel est CRITIQUE)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners2)
                
                count += 1
                print(f"   Frame {count} enregistrée")
            else:
                print("   ⚠️ Pas de damier trouvé, impossible de capturer.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if count < 10:
        print("❌ Pas assez de données (besoin de >10 frames). Sortie.")
        return

    # 2. Calibrer (Le Modèle "Plumb Bob")
    print("\n🧮 Calcul en cours... (Cela peut prendre un moment)")
    
    # Cette fonction utilise par défaut le modèle à 5 paramètres (k1, k2, p1, p2, k3)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # 3. Afficher les Résultats
    print("\n✅ CALIBRATION RÉUSSIE !")
    print(f"   Erreur de Re-projection : {ret:.4f} (Devrait être < 0.5)")
    print("\n--- MATRICE CAMÉRA (Intrinsèques) ---")
    print(f"fx: {mtx[0,0]:.2f}, fy: {mtx[1,1]:.2f}")
    print(f"cx: {mtx[0,2]:.2f}, cy: {mtx[1,2]:.2f}")
    
    print("\n--- COEFFICIENTS DE DISTORSION (Plumb Bob) ---")
    print("Ordre : [k1, k2, p1, p2, k3]")
    print(dist.ravel())
    
    # 4. Générer une sortie style YAML
    print("\n--- COPIER DANS VOTRE CONFIG ---")
    print("camera_matrix:")
    print(f"  data: {mtx.flatten().tolist()}")
    print("distortion_model: plumb_bob")
    print("distortion_coefficients:")
    print(f"  data: {dist.flatten().tolist()}")

if __name__ == "__main__":
    run_calibration()