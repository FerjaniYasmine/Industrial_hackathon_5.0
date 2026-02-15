"""
MODÈLE D'INTELLIGENCE ARTIFICIELLE - DÉTECTION DE FUITES
=========================================================
Modèle IA pour détecter et comprendre les fuites d'air comprimé
avec Random Forest et analyse intelligente des capteurs

États détectés:
- 0: Normal (Vert) ✅
- 1: Pré-fuite / Alarme Orange ⚠️
- 2: Fuite Critique / Alarme Rouge 🚨
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
import pickle
import json
from datetime import datetime

class ModeleDetectionFuites:
    """
    Modèle d'Intelligence Artificielle pour détecter les fuites d'air comprimé
    """
    
    def __init__(self, type_modele='random_forest'):
        """
        Initialise le modèle IA
        
        Args:
            type_modele: 'random_forest' ou 'gradient_boosting'
        """
        self.type_modele = type_modele
        self.modele = None
        self.scaler = StandardScaler()
        self.noms_features = None
        self.historique_entrainement = {}
        
        # Labels des états
        self.labels_etats = {
            0: 'Normal (Vert)',
            1: 'Pré-fuite - Alarme Orange',
            2: 'Fuite Critique - Alarme Rouge'
        }
        
        # Seuils de détection pour analyse
        self.seuils_normaux = {
            'vibration_hz': (20, 30),
            'pression_bar': (6.8, 7.2),
            'debit_m3h': (140, 160),
            'temperature_c': (32, 38),
            'ultrasons_db': (40, 50)
        }
        
        print(f"🤖 Modèle IA initialisé: {type_modele}")
    
    def charger_dataset(self, fichier_csv='dataset_fuites_air_comprime.csv'):
        """
        Charge le dataset depuis un fichier CSV
        
        Args:
            fichier_csv: Chemin vers le fichier CSV
        
        Returns:
            DataFrame pandas
        """
        print(f"\n📂 Chargement du dataset: {fichier_csv}")
        
        df = pd.read_csv(fichier_csv)
        
        print(f"✅ Dataset chargé: {len(df)} échantillons")
        print(f"   Colonnes: {list(df.columns)}")
        print(f"\n📊 Distribution des états:")
        for etat, count in df['etat'].value_counts().sort_index().items():
            pct = (count / len(df)) * 100
            print(f"   État {etat} ({self.labels_etats[etat]}): {count} ({pct:.1f}%)")
        
        return df
    
    def preparer_donnees(self, df):
        """
        Prépare les données pour l'entraînement
        
        Args:
            df: DataFrame avec les données
        
        Returns:
            X, y: Features et cibles
        """
        print(f"\n🔧 Préparation des données...")
        
        # Colonnes des features (capteurs)
        colonnes_features = ['vibration_hz', 'pression_bar', 'debit_m3h', 
                            'temperature_c', 'ultrasons_db']
        
        # Vérifier que toutes les colonnes existent
        for col in colonnes_features:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante: {col}")
        
        # Extraire X (features) et y (cible)
        X = df[colonnes_features].values
        y = df['etat'].values
        
        self.noms_features = colonnes_features
        
        print(f"✅ Données préparées:")
        print(f"   Features (X): {X.shape}")
        print(f"   Cibles (y): {y.shape}")
        print(f"   Features utilisées: {colonnes_features}")
        
        return X, y
    
    def entrainer(self, df, test_size=0.2, validation_croisee=True):
        """
        Entraîne le modèle IA sur le dataset
        
        Args:
            df: DataFrame avec les données
            test_size: Proportion du test set (0.2 = 20%)
            validation_croisee: Effectuer validation croisée
        
        Returns:
            dict avec les résultats d'entraînement
        """
        print("\n" + "=" * 70)
        print("   ENTRAÎNEMENT DU MODÈLE D'INTELLIGENCE ARTIFICIELLE")
        print("=" * 70)
        
        # Préparer les données
        X, y = self.preparer_donnees(df)
        
        # Split train/test avec stratification (garde la même proportion d'états)
        print(f"\n📊 Séparation des données (test={test_size*100:.0f}%)...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42, 
            stratify=y  # Garde les mêmes proportions
        )
        
        print(f"   Train set: {len(X_train)} échantillons")
        print(f"   Test set:  {len(X_test)} échantillons")
        
        # Normalisation des données (très important!)
        print(f"\n🔄 Normalisation des données...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print(f"   ✓ Normalisation appliquée (StandardScaler)")
        
        # Créer le modèle
        print(f"\n🧠 Création du modèle: {self.type_modele}")
        
        if self.type_modele == 'random_forest':
            self.modele = RandomForestClassifier(
                n_estimators=200,        # 200 arbres de décision
                max_depth=15,            # Profondeur max
                min_samples_split=5,     # Min échantillons pour split
                min_samples_leaf=2,      # Min échantillons par feuille
                max_features='sqrt',     # Features à considérer par split
                random_state=42,
                class_weight='balanced', # Équilibrer les classes
                n_jobs=-1               # Utiliser tous les CPU
            )
        elif self.type_modele == 'gradient_boosting':
            self.modele = GradientBoostingClassifier(
                n_estimators=150,
                max_depth=7,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Type de modèle non supporté: {self.type_modele}")
        
        print(f"   Paramètres: {self.modele.get_params()}")
        
        # Entraînement
        print(f"\n⏳ Entraînement en cours...")
        debut = datetime.now()
        self.modele.fit(X_train_scaled, y_train)
        duree = (datetime.now() - debut).total_seconds()
        print(f"✅ Entraînement terminé en {duree:.2f} secondes")
        
        # Validation croisée
        if validation_croisee:
            print(f"\n🔄 Validation croisée (5-fold)...")
            scores_cv = cross_val_score(
                self.modele, X_train_scaled, y_train, 
                cv=5, scoring='accuracy'
            )
            print(f"   Scores: {scores_cv}")
            print(f"   Moyenne: {scores_cv.mean():.4f} (+/- {scores_cv.std() * 2:.4f})")
        
        # Prédictions sur le test set
        print(f"\n📊 Évaluation sur le test set...")
        y_pred = self.modele.predict(X_test_scaled)
        y_pred_proba = self.modele.predict_proba(X_test_scaled)
        
        # Calculer les métriques
        precision_globale = accuracy_score(y_test, y_pred)
        precision, rappel, f1, support = precision_recall_fscore_support(
            y_test, y_pred, average=None
        )
        
        # Afficher les résultats
        print(f"\n" + "=" * 70)
        print(f"   RÉSULTATS D'ENTRAÎNEMENT")
        print(f"=" * 70)
        print(f"\n🎯 Précision globale: {precision_globale:.2%}")
        
        print(f"\n📊 Rapport de classification détaillé:")
        print(classification_report(
            y_test, y_pred, 
            target_names=list(self.labels_etats.values()),
            digits=4
        ))
        
        print(f"\n🔢 Matrice de confusion:")
        cm = confusion_matrix(y_test, y_pred)
        print(f"                 Prédit→")
        print(f"    Réel↓     Normal  Pré-fuite  Critique")
        for i, row in enumerate(cm):
            label = ['Normal', 'Pré-fuite', 'Critique'][i]
            print(f"    {label:10} {row[0]:6} {row[1]:10} {row[2]:8}")
        
        # Importance des features
        if hasattr(self.modele, 'feature_importances_'):
            print(f"\n⭐ Importance des capteurs:")
            importances = pd.DataFrame({
                'capteur': self.noms_features,
                'importance': self.modele.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for idx, row in importances.iterrows():
                bar = '█' * int(row['importance'] * 50)
                print(f"   {row['capteur']:20} {row['importance']:.3f} {bar}")
        
        # Sauvegarder l'historique
        self.historique_entrainement = {
            'date': datetime.now().isoformat(),
            'type_modele': self.type_modele,
            'precision_globale': float(precision_globale),
            'precision_par_classe': precision.tolist(),
            'rappel_par_classe': rappel.tolist(),
            'f1_par_classe': f1.tolist(),
            'matrice_confusion': cm.tolist(),
            'importance_features': importances.to_dict('records') if hasattr(self.modele, 'feature_importances_') else None,
            'validation_croisee': scores_cv.tolist() if validation_croisee else None
        }
        
        print(f"\n" + "=" * 70)
        print(f"✅ ENTRAÎNEMENT RÉUSSI!")
        print(f"=" * 70)
        
        return {
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'precision_globale': precision_globale,
            'historique': self.historique_entrainement
        }
    
    def predire(self, donnees_capteurs):
        """
        Prédit l'état du système à partir des données capteurs
        
        Args:
            donnees_capteurs: dict ou array avec les valeurs des capteurs
                            {'vibration_hz': 25, 'pression_bar': 7.0, ...}
        
        Returns:
            dict avec la prédiction, probabilités et analyse
        """
        if self.modele is None:
            raise ValueError("Le modèle n'est pas encore entraîné! Appelez entrainer() d'abord.")
        
        # Convertir en array si dict
        if isinstance(donnees_capteurs, dict):
            X = np.array([[
                donnees_capteurs['vibration_hz'],
                donnees_capteurs['pression_bar'],
                donnees_capteurs['debit_m3h'],
                donnees_capteurs['temperature_c'],
                donnees_capteurs['ultrasons_db']
            ]])
        else:
            X = np.array([donnees_capteurs])
        
        # Normaliser
        X_scaled = self.scaler.transform(X)
        
        # Prédire
        etat_predit = self.modele.predict(X_scaled)[0]
        probabilites = self.modele.predict_proba(X_scaled)[0]
        
        # Analyser les capteurs
        analyse = self._analyser_capteurs(donnees_capteurs if isinstance(donnees_capteurs, dict) else {
            'vibration_hz': X[0][0],
            'pression_bar': X[0][1],
            'debit_m3h': X[0][2],
            'temperature_c': X[0][3],
            'ultrasons_db': X[0][4]
        })
        
        # Générer recommandation
        recommandations = {
            0: "✅ Système fonctionne normalement. Continuer surveillance de routine.",
            1: "⚠️ ALARME ORANGE: Anomalies détectées. Inspection recommandée dans les 48h pour éviter une fuite majeure.",
            2: "🚨 ALARME ROUGE: Fuite critique détectée! Intervention urgente requise. Risque de perte énergétique importante."
        }
        
        # Niveau de risque
        niveaux_risque = {
            0: 'FAIBLE',
            1: 'MOYEN',
            2: 'CRITIQUE'
        }
        
        return {
            'etat': int(etat_predit),
            'label': self.labels_etats[etat_predit],
            'niveau_risque': niveaux_risque[etat_predit],
            'probabilites': {
                'normal': float(probabilites[0]),
                'pre_fuite': float(probabilites[1]),
                'fuite_critique': float(probabilites[2])
            },
            'confiance': float(max(probabilites)),
            'recommandation': recommandations[etat_predit],
            'analyse_capteurs': analyse,
            'capteurs': donnees_capteurs if isinstance(donnees_capteurs, dict) else None
        }
    
    def _analyser_capteurs(self, capteurs):
        """
        Analyse intelligente des valeurs des capteurs
        
        Args:
            capteurs: dict avec les valeurs
        
        Returns:
            dict avec l'analyse de chaque capteur
        """
        analyse = {}
        
        for capteur, valeur in capteurs.items():
            if capteur in self.seuils_normaux:
                min_normal, max_normal = self.seuils_normaux[capteur]
                
                if valeur < min_normal:
                    statut = 'BAS'
                    anomalie = True
                elif valeur > max_normal:
                    statut = 'HAUT'
                    anomalie = True
                else:
                    statut = 'NORMAL'
                    anomalie = False
                
                ecart_pct = abs(valeur - (min_normal + max_normal) / 2) / ((max_normal - min_normal) / 2) * 100
                
                analyse[capteur] = {
                    'valeur': float(valeur),
                    'statut': statut,
                    'anomalie': anomalie,
                    'ecart_pct': float(ecart_pct),
                    'plage_normale': (min_normal, max_normal)
                }
        
        return analyse
    
    def comprendre_prediction(self, prediction):
        """
        Explique en détail pourquoi le modèle a fait cette prédiction
        
        Args:
            prediction: résultat de predire()
        
        Returns:
            str: explication détaillée
        """
        explication = []
        
        explication.append(f"\n{'='*70}")
        explication.append(f"   ANALYSE INTELLIGENTE DE LA PRÉDICTION")
        explication.append(f"{'='*70}")
        
        explication.append(f"\n🎯 RÉSULTAT: {prediction['label']}")
        explication.append(f"   Niveau de risque: {prediction['niveau_risque']}")
        explication.append(f"   Confiance du modèle: {prediction['confiance']:.1%}")
        
        explication.append(f"\n💭 PROBABILITÉS:")
        explication.append(f"   • Normal:         {prediction['probabilites']['normal']:.1%}")
        explication.append(f"   • Pré-fuite:      {prediction['probabilites']['pre_fuite']:.1%}")
        explication.append(f"   • Fuite critique: {prediction['probabilites']['fuite_critique']:.1%}")
        
        explication.append(f"\n🔍 ANALYSE DES CAPTEURS:")
        capteurs_anormaux = []
        
        for capteur, analyse in prediction['analyse_capteurs'].items():
            symbole = '❌' if analyse['anomalie'] else '✅'
            explication.append(f"   {symbole} {capteur}: {analyse['valeur']:.2f} ({analyse['statut']})")
            
            if analyse['anomalie']:
                capteurs_anormaux.append(capteur)
                explication.append(f"      ⚠️ Écart: {analyse['ecart_pct']:.1f}% de la normale")
                explication.append(f"      📊 Plage normale: {analyse['plage_normale'][0]}-{analyse['plage_normale'][1]}")
        
        if capteurs_anormaux:
            explication.append(f"\n⚠️ CAPTEURS ANORMAUX DÉTECTÉS:")
            for capteur in capteurs_anormaux:
                explication.append(f"   • {capteur}")
        else:
            explication.append(f"\n✅ Tous les capteurs sont dans les plages normales")
        
        explication.append(f"\n💡 {prediction['recommandation']}")
        
        if prediction['etat'] == 1:
            explication.append(f"\n🔧 ACTIONS PRÉVENTIVES RECOMMANDÉES:")
            explication.append(f"   1. Inspecter les connexions et tuyaux")
            explication.append(f"   2. Vérifier l'étanchéité des raccords")
            explication.append(f"   3. Tester avec détecteur ultrason")
            explication.append(f"   4. Planifier maintenance dans 48h")
        
        elif prediction['etat'] == 2:
            explication.append(f"\n🚨 ACTIONS URGENTES REQUISES:")
            explication.append(f"   1. ⚠️ Localiser la fuite immédiatement")
            explication.append(f"   2. 🔧 Isoler la zone si possible")
            explication.append(f"   3. 📞 Appeler l'équipe de maintenance")
            explication.append(f"   4. 💰 Estimer le coût de la fuite (énergie perdue)")
            explication.append(f"   5. 📋 Documenter l'incident")
        
        explication.append(f"\n{'='*70}")
        
        return '\n'.join(explication)
    
    def sauvegarder(self, fichier='modele_ia_fuites.pkl'):
        """
        Sauvegarde le modèle entraîné
        
        Args:
            fichier: Nom du fichier de sortie
        """
        if self.modele is None:
            raise ValueError("Aucun modèle à sauvegarder!")
        
        donnees_modele = {
            'modele': self.modele,
            'scaler': self.scaler,
            'noms_features': self.noms_features,
            'type_modele': self.type_modele,
            'labels_etats': self.labels_etats,
            'seuils_normaux': self.seuils_normaux,
            'historique': self.historique_entrainement
        }
        
        with open(fichier, 'wb') as f:
            pickle.dump(donnees_modele, f)
        
        print(f"\n💾 Modèle sauvegardé: {fichier}")
        print(f"   Type: {self.type_modele}")
        print(f"   Précision: {self.historique_entrainement.get('precision_globale', 0):.2%}")
    
    @classmethod
    def charger(cls, fichier='modele_ia_fuites.pkl'):
        """
        Charge un modèle sauvegardé
        
        Args:
            fichier: Chemin vers le fichier
        
        Returns:
            Instance du modèle chargé
        """
        with open(fichier, 'rb') as f:
            donnees_modele = pickle.load(f)
        
        instance = cls(type_modele=donnees_modele['type_modele'])
        instance.modele = donnees_modele['modele']
        instance.scaler = donnees_modele['scaler']
        instance.noms_features = donnees_modele['noms_features']
        instance.labels_etats = donnees_modele['labels_etats']
        instance.seuils_normaux = donnees_modele['seuils_normaux']
        instance.historique_entrainement = donnees_modele['historique']
        
        print(f"✅ Modèle chargé: {fichier}")
        print(f"   Type: {instance.type_modele}")
        print(f"   Précision: {instance.historique_entrainement.get('precision_globale', 0):.2%}")
        
        return instance


def main():
    """
    Fonction principale - Entraîne et teste le modèle
    """
    
    print("=" * 70)
    print("   MODÈLE D'INTELLIGENCE ARTIFICIELLE")
    print("   Détection et Compréhension des Fuites d'Air Comprimé")
    print("=" * 70)
    
    # 1. Créer le modèle
    print("\n🤖 Étape 1: Initialisation du modèle IA")
    modele = ModeleDetectionFuites(type_modele='random_forest')
    
    # 2. Charger le dataset
    print("\n📂 Étape 2: Chargement des données")
    df = modele.charger_dataset('dataset_fuites_air_comprime.csv')
    
    # 3. Entraîner le modèle
    print("\n🎓 Étape 3: Entraînement du modèle")
    resultats = modele.entrainer(df, test_size=0.2, validation_croisee=True)
    
    # 4. Sauvegarder le modèle
    print("\n💾 Étape 4: Sauvegarde du modèle")
    modele.sauvegarder('modele_ia_fuites.pkl')
    
    # 5. Tests de prédiction
    print("\n" + "=" * 70)
    print("   TESTS DE PRÉDICTION - COMPRÉHENSION DES FUITES")
    print("=" * 70)
    
    scenarios_test = [
        {
            'nom': '✅ Système Normal - Tout va bien',
            'capteurs': {
                'vibration_hz': 24.5,
                'pression_bar': 7.05,
                'debit_m3h': 148.0,
                'temperature_c': 34.5,
                'ultrasons_db': 43.2
            }
        },
        {
            'nom': '⚠️ Pré-fuite Détectée - Attention requise',
            'capteurs': {
                'vibration_hz': 33.8,
                'pression_bar': 6.45,
                'debit_m3h': 167.5,
                'temperature_c': 39.2,
                'ultrasons_db': 59.8
            }
        },
        {
            'nom': '🚨 Fuite Critique - Urgence!',
            'capteurs': {
                'vibration_hz': 46.2,
                'pression_bar': 5.75,
                'debit_m3h': 196.3,
                'temperature_c': 44.8,
                'ultrasons_db': 77.5
            }
        }
    ]
    
    for scenario in scenarios_test:
        print(f"\n{'▬'*70}")
        print(f"TEST: {scenario['nom']}")
        print(f"{'▬'*70}")
        
        print(f"\n📊 Lectures des capteurs:")
        for capteur, valeur in scenario['capteurs'].items():
            print(f"   • {capteur}: {valeur}")
        
        # Faire la prédiction
        prediction = modele.predire(scenario['capteurs'])
        
        # Afficher l'explication complète
        explication = modele.comprendre_prediction(prediction)
        print(explication)
    
    print("\n" + "=" * 70)
    print("✅ ENTRAÎNEMENT ET TESTS TERMINÉS AVEC SUCCÈS!")
    print("=" * 70)
    
    print(f"\n📊 Résumé:")
    print(f"   • Modèle entraîné: Random Forest")
    print(f"   • Précision: {resultats['precision_globale']:.2%}")
    print(f"   • Fichier sauvegardé: modele_ia_fuites.pkl")
    print(f"   • Prêt pour déploiement en production!")
    
    return modele, resultats


if __name__ == "__main__":
    modele, resultats = main()
