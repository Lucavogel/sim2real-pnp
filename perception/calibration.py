import cv2
import numpy as np
import glob

# Dimensions de l'échiquier (nombre de coins intérieurs)
chessboard_size = (7, 6)  # 7 coins par ligne, 6 par colonne
square_size = 0.024  # taille d'une case en mètres ou en cm

# Critère de terminaison pour la précision subpix
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Préparation des points 3D (ex: (0,0,0), (1,0,0), ..., (6,5,0))
objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Stockage des points 3D et 2D
objpoints = []  # Points 3D dans le monde réel
imgpoints = []  # Points 2D dans l'image

cap = cv2.VideoCapture(0)  # 0 = première caméra USB ou PiCamera simulée

if not cap.isOpened():
    print("Erreur : caméra non détectée.")
    exit()

print("Appuie sur 'c' pour capturer une image, 'q' pour quitter")

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

    cv2.imshow('Calibration', display)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('c') and ret_corners:
        print("→ Image capturée")
        objpoints.append(objp)
        imgpoints.append(corners_subpix)
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Calibrage de la caméra
if len(objpoints) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray.shape[::-1], None, None
    )

    print("\n=== Résultats du calibrage ===")
    print("Matrice de la caméra :\n", camera_matrix)
    print("Coefficients de distorsion :\n", dist_coeffs)

    # Sauvegarde dans un fichier
    np.savez("calibration_data.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

else:
    print("Aucune image valide n'a été capturée pour le calibrage.")
