import cv2
import numpy as np
import glob
import pickle
import time

# Dimensions de l'échiquier (nombre de coins intérieurs)
chessboard_size = (8, 6)  # 9 coins par ligne, 6 par colonne
square_size = 0.024  # taille d'une case en mètres (IMPORTANT: mesurez après impression!)

print(f"⚠️ IMPORTANT: Vérifiez que vos carrés font bien {square_size*1000:.1f}mm")
print("Mesurez avec une règle après impression !")

# Critère de terminaison pour la précision subpix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparation des points 3D (ex: (0,0,0), (1,0,0), ..., (6,5,0))
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Stockage des points 3D et 2D
objpoints = []  # Points 3D dans le monde réel
imgpoints = []  # Points 2D dans l'image

cap = cv2.VideoCapture(4)  # 0 = première caméra USB ou PiCamera simulée

if not cap.isOpened():
    print("Erreur : caméra non détectée.")
    exit()

print("📸 Instructions de calibration:")
print("- Appuyez sur 'c' pour capturer une image quand le damier est détecté")
print("- Bougez le damier sous différents angles et distances")
print("- Capturez 15-20 images minimum")
print("- Appuyez sur 'q' pour terminer")
print("- Couvrez toute la zone de l'image avec le damier")

num_captured = 0
min_images = 15

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détection des coins
    ret_corners, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    # Affichage des coins trouvés
    display = frame.copy()
    if ret_corners:
        corners_subpix = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        cv2.drawChessboardCorners(display, chessboard_size, corners_subpix, ret_corners)
        cv2.putText(display, "DAMIER DETECTE - 'c' pour capturer", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Damier non detecte", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Affichage du compteur
    color = (0, 255, 0) if num_captured >= min_images else (255, 255, 255)
    cv2.putText(display, f"Images: {num_captured}/{min_images} min", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow('Calibration - c: capturer, q: quitter', display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and ret_corners:
        # Améliore la précision des coins
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        num_captured += 1
        print(f"✅ Image {num_captured} capturée")
        
        if num_captured >= min_images:
            print("🎯 Nombre d'images suffisant pour une bonne calibration!")
            
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibrage de la caméra
if len(objpoints) >= 5:
    print(f"\n🎯 Calibration avec {len(objpoints)} images...")
    
    # Vérification du nombre d'images
    if len(objpoints) < 10:
        print(f"⚠️ Seulement {len(objpoints)} images capturées")
        print("Recommandé : 15-20 images minimum pour une bonne calibration")
        
    # Calibration avec flags pour plus de robustesse
    flags = cv2.CALIB_RATIONAL_MODEL + cv2.CALIB_THIN_PRISM_MODEL
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None, flags=flags
    )

    if ret:
        # Calcule l'erreur de reprojection
        total_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error

        mean_error = total_error / len(objpoints)

        print("\n✅ === RÉSULTATS DE CALIBRATION ===")
        print(f"Erreur moyenne de reprojection: {mean_error:.3f} pixels")
        
        if mean_error < 0.5:
            print("✅ Excellente précision (< 0.5 pixels)")
        elif mean_error < 1.0:
            print("✅ Bonne précision (< 1.0 pixel)")
        else:
            print("⚠️ Précision moyenne - recommandé de refaire avec plus d'images")
            
        print(f"Distance focale: fx={camera_matrix[0,0]:.1f}, fy={camera_matrix[1,1]:.1f}")
        print(f"Centre optique: cx={camera_matrix[0,2]:.1f}, cy={camera_matrix[1,2]:.1f}")
        print(f"Nombre d'images utilisées: {len(objpoints)}")

        # Sauvegarde améliorée
        calibration_data = {
            'camera_matrix': camera_matrix,
            'dist_coeffs': dist_coeffs,
            'chessboard_size': chessboard_size,
            'square_size': square_size,
            'mean_error': mean_error,
            'num_images': len(objpoints),
            'timestamp': time.time(),
            'opencv_version': cv2.__version__
        }
        
        # Sauvegarde en format pickle (moderne)
        with open("camera_calibration.pkl", 'wb') as f:
            pickle.dump(calibration_data, f)
        
        # Sauvegarde aussi en format npz (compatible avec ancien code)
        np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
        
        print("💾 Calibration sauvegardée dans:")
        print("   - camera_calibration.pkl (format moderne)")
        print("   - calibration_data.npz (format legacy)")
        print("Le fichier .pkl sera utilisé automatiquement par la détection AprilTags")
        
    else:
        print("❌ Échec de la calibration")
        
else:
    print(f"❌ Pas assez d'images pour calibration (minimum 5, capturé: {len(objpoints)})")
    print("Relancez le script et capturez plus d'images")
