"""
COMPUTER VISION - Detection de Fuites par Couleur
==================================================
Ce module active la camera pour localiser visuellement
les fuites en detectant des marqueurs jaunes

La camera reste OUVERTE jusqu'a retour a l'etat NORMAL
"""

import cv2
import numpy as np
from PIL import Image

def get_limits(color):
    """
    Calcule les seuils de couleur pour le masque HSV
    
    Args:
        color: Couleur BGR [B, G, R]
    
    Returns:
        lowerLimit, upperLimit pour cv2.inRange()
    """
    c = np.uint8([[color]]) 
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    # Limites basse et haute (Teinte +/- 10)
    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit


def run_cv(stop_flag):
    """
    Lance la detection visuelle de fuites par couleur
    
    Cette fonction tourne EN CONTINU jusqu'a ce que stop_flag devienne True
    (c'est-a-dire quand l'etat redevient NORMAL)
    
    Args:
        stop_flag: threading.Event() qui signale l'arret quand etat = NORMAL
    """
    print("\n" + "="*70)
    print("   COMPUTER VISION ACTIVEE - LOCALISATION DE FUITE")
    print("="*70)
    print("\nRecherche de marqueurs JAUNES...")
    print("La camera restera ouverte jusqu'au retour a l'etat NORMAL")
    print("Appuyez sur 'q' pour arreter manuellement\n")
    
    # Couleur cible: JAUNE [Bleu=0, Vert=255, Rouge=255]
    couleur_cible = [0, 255, 255]
    
    # Ouvrir la webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERREUR: Impossible d'acceder a la camera!")
        print("Verifiez que votre webcam est branchee.")
        return
    
    print("Camera activee avec succes")
    
    # Compteur
    import time
    debut = time.time()
    fuite_detectee = False
    
    try:
        # Boucle INFINIE jusqu'a ce que stop_flag soit active
        while not stop_flag.is_set():
            # Lire l'image
            ret, frame = cap.read()
            
            if not ret:
                print("ERREUR: Impossible de lire l'image de la camera")
                break
            
            # Calculer temps ecoule
            temps_ecoule = int(time.time() - debut)
            
            # Convertir en HSV
            hsvImage = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Obtenir les limites
            lowerLimit, upperLimit = get_limits(color=couleur_cible)
            
            # Creer le masque
            mask = cv2.inRange(hsvImage, lowerLimit, upperLimit)
            
            # Localiser les pixels
            mask_pillow = Image.fromarray(mask)
            bbox = mask_pillow.getbbox()
            
            # Afficher le temps ecoule
            cv2.putText(frame, f"Temps: {temps_ecoule}s", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Afficher statut
            cv2.putText(frame, "EN ATTENTE RETOUR ETAT NORMAL", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # Si on trouve l'objet jaune
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                
                # Rectangle vert autour de la zone
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)
                
                # Texte d'alerte
                cv2.putText(frame, "FUITE DETECTEE ICI !", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                
                # Coordonnees
                centre_x = (x1 + x2) // 2
                centre_y = (y1 + y2) // 2
                cv2.circle(frame, (centre_x, centre_y), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"Position: ({centre_x}, {centre_y})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if not fuite_detectee:
                    fuite_detectee = True
                    print(f"\nFUITE LOCALISEE!")
                    print(f"  Position: X={centre_x}, Y={centre_y}")
                    print(f"  Zone: ({x1},{y1}) -> ({x2},{y2})")
            
            # Afficher l'image
            cv2.imshow('Detection Visuelle de Fuite', frame)
            
            # Quitter avec 'q' (option manuelle)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nDetection arretee manuellement")
                break
    
    except Exception as e:
        print(f"\nERREUR dans la detection visuelle: {e}")
    
    finally:
        # Nettoyer
        cap.release()
        cv2.destroyAllWindows()
        
        print("\n" + "="*70)
        if fuite_detectee:
            print("   RESULTAT: Fuite localisee visuellement")
        else:
            print("   RESULTAT: Aucune fuite visible (marqueur jaune)")
        print("="*70)
        print("Computer Vision desactivee - Retour a l'etat NORMAL\n")


def test_camera():
    """Fonction de test simple pour verifier la camera"""
    print("\nTest de la camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("ERREUR: Camera non accessible!")
        return False
    
    print("Camera OK!")
    print("Appuyez sur 'q' pour fermer")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Test Camera', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Test termine")
    return True


if __name__ == "__main__":
    # Test si execute directement
    print("="*70)
    print("   TEST DU MODULE COMPUTER VISION")
    print("="*70)
    
    choix = input("\n1. Tester la camera\n2. Tester la detection de couleur (10s)\nChoix (1 ou 2): ")
    
    if choix == "1":
        test_camera()
    else:
        import threading
        stop_event = threading.Event()
        
        # Lancer la detection
        thread = threading.Thread(target=run_cv, args=(stop_event,))
        thread.start()
        
        # Arreter apres 10 secondes (pour le test)
        import time
        time.sleep(10)
        print("\nArret du test...")
        stop_event.set()
        thread.join()
        print("Test termine")
