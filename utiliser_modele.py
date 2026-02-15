"""
SYSTEME HYBRIDE - IA + COMPUTER VISION
=======================================
Ce script combine:
1. Modele IA pour detecter les fuites (capteurs)
2. Computer Vision pour localiser visuellement (camera)

LOGIQUE:
- Camera FERMEE si etat = NORMAL
- Camera S'OUVRE si etat = PRE-FUITE ou FUITE CRITIQUE
- Camera RESTE OUVERTE jusqu'au retour a l'etat NORMAL
"""

import time
import threading
from modele_ia_fuites import ModeleDetectionFuites
from computer_vision import run_cv
import random

# Configuration
INTERVALLE_LECTURE = 5  # Secondes entre chaque lecture capteurs

# --- Charger le modele IA ---
print("="*70)
print("   SYSTEME HYBRIDE DE DETECTION DE FUITES")
print("="*70)
print("\nChargement du modele IA...")

try:
    modele = ModeleDetectionFuites.charger('modele_ia_fuites.pkl')
    print("Modele charge avec succes!")
except FileNotFoundError:
    print("\nERREUR: Modele non trouve!")
    print("Executez d'abord: python modele_ia_fuites.py")
    exit()

print("\n" + "="*70)
print("   SYSTEME PRET - SURVEILLANCE ACTIVE")
print("="*70)
print(f"\nConfiguration:")
print(f"  - Lecture capteurs: toutes les {INTERVALLE_LECTURE}s")
print(f"  - Computer Vision: OUVERTE jusqu'a retour NORMAL")
print(f"  - Appuyez sur Ctrl+C pour arreter")
print("\n" + "="*70)

# Variables pour gerer la Computer Vision
cv_thread = None
stop_cv = None  # Event pour arreter la CV
etat_precedent = 0  # Tracker l'etat precedent

# Compteur de lectures
compteur_lectures = 0

try:
    while True:
        compteur_lectures += 1
        
        print(f"\n[Lecture #{compteur_lectures}] {time.strftime('%H:%M:%S')}")
        print("-" * 70)
        
        # --- SIMULATION LECTURE CAPTEURS ---
        # En production, remplacer par vraies lectures capteurs
        
        # Simuler differents etats
        etat_simule = random.choices([0, 1, 2], weights=[0.7, 0.2, 0.1])[0]
        
        if etat_simule == 0:  # Normal
            capteurs = {
                'vibration_hz': random.gauss(25, 3),
                'pression_bar': random.gauss(7.0, 0.2),
                'debit_m3h': random.gauss(150, 10),
                'temperature_c': random.gauss(35, 3),
                'ultrasons_db': random.gauss(45, 4)
            }
        elif etat_simule == 1:  # Pre-fuite
            capteurs = {
                'vibration_hz': random.gauss(32, 4),
                'pression_bar': random.gauss(6.5, 0.3),
                'debit_m3h': random.gauss(165, 12),
                'temperature_c': random.gauss(38, 4),
                'ultrasons_db': random.gauss(58, 6)
            }
        else:  # Fuite critique
            capteurs = {
                'vibration_hz': random.gauss(45, 6),
                'pression_bar': random.gauss(5.8, 0.5),
                'debit_m3h': random.gauss(195, 15),
                'temperature_c': random.gauss(43, 5),
                'ultrasons_db': random.gauss(75, 8)
            }
        
        # Afficher les lectures
        print("Capteurs:")
        for nom, valeur in capteurs.items():
            print(f"  {nom:20s} {valeur:6.2f}")
        
        # --- PREDICTION IA ---
        prediction = modele.predire(capteurs)
        etat_actuel = prediction['etat']
        
        print(f"\nPrediction IA:")
        print(f"  Etat: {prediction['label']}")
        print(f"  Confiance: {prediction['confiance']*100:.1f}%")
        print(f"  Risque: {prediction['niveau_risque']}")
        
        # --- LOGIQUE D'OUVERTURE/FERMETURE CAMERA ---
        
        # CAS 1: PASSAGE DE NORMAL A ANORMAL -> OUVRIR CAMERA
        if etat_precedent == 0 and etat_actuel in [1, 2]:
            print("\nTransition: NORMAL -> ANORMAL")
            print("  Computer Vision: OUVERTURE DE LA CAMERA...")
            
            # Creer un event d'arret
            stop_cv = threading.Event()
            
            # Lancer la CV dans un thread
            cv_thread = threading.Thread(
                target=run_cv,
                args=(stop_cv,),
                daemon=True
            )
            cv_thread.start()
            print("  Camera activee!")
        
        # CAS 2: RESTE ANORMAL -> CAMERA RESTE OUVERTE
        elif etat_precedent in [1, 2] and etat_actuel in [1, 2]:
            print("\nEtat: TOUJOURS ANORMAL")
            print("  Computer Vision: RESTE OUVERTE")
            if cv_thread and cv_thread.is_alive():
                print("  Camera toujours active...")
            else:
                print("  ATTENTION: Thread CV mort, relance...")
                stop_cv = threading.Event()
                cv_thread = threading.Thread(
                    target=run_cv,
                    args=(stop_cv,),
                    daemon=True
                )
                cv_thread.start()
        
        # CAS 3: PASSAGE D'ANORMAL A NORMAL -> FERMER CAMERA
        elif etat_precedent in [1, 2] and etat_actuel == 0:
            print("\nTransition: ANORMAL -> NORMAL")
            print("  Computer Vision: FERMETURE DE LA CAMERA...")
            
            if stop_cv:
                stop_cv.set()  # Signaler l'arret
                print("  Signal d'arret envoye")
                
                # Attendre que le thread se termine
                if cv_thread and cv_thread.is_alive():
                    cv_thread.join(timeout=2)
                    if cv_thread.is_alive():
                        print("  ATTENTION: Thread CV ne repond pas")
                    else:
                        print("  Camera fermee avec succes")
            
            stop_cv = None
            cv_thread = None
        
        # CAS 4: RESTE NORMAL -> CAMERA RESTE FERMEE
        else:  # etat_precedent == 0 and etat_actuel == 0
            print("\nEtat: NORMAL")
            print("  Computer Vision: FERMEE")
        
        # Afficher recommandation
        print(f"\nRecommandation:")
        print(f"  {prediction['recommandation']}")
        
        # Mettre a jour l'etat precedent
        etat_precedent = etat_actuel
        
        # Afficher statut CV
        print(f"\nStatut Computer Vision:")
        if cv_thread and cv_thread.is_alive():
            print("  [ACTIVE] Camera ouverte")
        else:
            print("  [INACTIVE] Camera fermee")
        
        # Attendre avant prochaine lecture
        print(f"\nProchaine lecture dans {INTERVALLE_LECTURE}s...")
        time.sleep(INTERVALLE_LECTURE)

except KeyboardInterrupt:
    print("\n\n" + "="*70)
    print("   ARRET DU SYSTEME PAR L'UTILISATEUR")
    print("="*70)
    
    # Arreter la CV si active
    if stop_cv:
        print("\nFermeture de la Computer Vision...")
        stop_cv.set()
        if cv_thread and cv_thread.is_alive():
            cv_thread.join(timeout=3)
    
    print(f"\nStatistiques:")
    print(f"  Lectures effectuees: {compteur_lectures}")
    print("\nSysteme arrete proprement.")

except Exception as e:
    print(f"\n\nERREUR CRITIQUE: {e}")
    print("Arret du systeme")
    
    # Arreter la CV si active
    if stop_cv:
        stop_cv.set()

finally:
    # Nettoyage final
    if stop_cv:
        stop_cv.set()
    if cv_thread and cv_thread.is_alive():
        print("\nAttente de l'arret de la Computer Vision...")
        cv_thread.join(timeout=5)
    
    print("\nSysteme completement arrete.")
