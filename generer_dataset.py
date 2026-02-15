"""
GÉNÉRATEUR DE DATASET - Détection de Fuites d'Air Comprimé
============================================================
Ce script génère uniquement les données synthétiques de capteurs
pour l'entraînement futur d'un modèle IA

Capteurs: Vibration, Pression, Débit, Température, Ultrasons
États: 0=Normal | 1=Pré-fuite (Orange) | 2=Fuite Critique (Rouge)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class GenerateurDatasetFuites:
    """Générateur de données synthétiques pour capteurs industriels"""
    
    def __init__(self, random_state=42):
        np.random.seed(random_state)
        
        # Paramètres pour système NORMAL (État 0)
        self.params_normal = {
            'vibration_hz': {'mean': 25, 'std': 3},      # Hz
            'pression_bar': {'mean': 7.0, 'std': 0.2},   # bar
            'debit_m3h': {'mean': 150, 'std': 10},       # m³/h
            'temperature_c': {'mean': 35, 'std': 3},     # °C
            'ultrasons_db': {'mean': 45, 'std': 4}       # dB
        }
        
        # Paramètres pour PRÉ-FUITE (État 1 - Alarme Orange)
        self.params_pre_fuite = {
            'vibration_hz': {'mean': 32, 'std': 4},      
            'pression_bar': {'mean': 6.5, 'std': 0.3},   
            'debit_m3h': {'mean': 165, 'std': 12},       
            'temperature_c': {'mean': 38, 'std': 4},     
            'ultrasons_db': {'mean': 58, 'std': 6}       
        }
        
        # Paramètres pour FUITE CRITIQUE (État 2 - Alarme Rouge)
        self.params_fuite_critique = {
            'vibration_hz': {'mean': 45, 'std': 6},      
            'pression_bar': {'mean': 5.8, 'std': 0.5},   
            'debit_m3h': {'mean': 195, 'std': 15},       
            'temperature_c': {'mean': 43, 'std': 5},     
            'ultrasons_db': {'mean': 75, 'std': 8}       
        }
    
    def generer_lecture_capteur(self, etat):
        """
        Génère une lecture des 5 capteurs selon l'état
        
        Args:
            etat: 0=Normal, 1=Pré-fuite, 2=Fuite critique
        
        Returns:
            dict avec les valeurs des capteurs
        """
        
        # Choisir les paramètres selon l'état
        if etat == 0:
            params = self.params_normal
        elif etat == 1:
            params = self.params_pre_fuite
        else:  # etat == 2
            params = self.params_fuite_critique
        
        # Générer les valeurs avec distribution normale
        lecture = {}
        for capteur, param in params.items():
            valeur = np.random.normal(param['mean'], param['std'])
            
            # Ajouter un peu de bruit réaliste
            bruit = np.random.normal(0, param['std'] * 0.1)
            valeur += bruit
            
            # Éviter les valeurs négatives
            lecture[capteur] = max(0, valeur)
        
        lecture['etat'] = etat
        
        return lecture
    
    def generer_dataset(self, 
                       n_normal=3000, 
                       n_pre_fuite=1500, 
                       n_fuite_critique=1500,
                       ajouter_timestamp=True):
        """
        Génère un dataset complet avec les 3 états
        
        Args:
            n_normal: Nombre d'échantillons normaux
            n_pre_fuite: Nombre d'échantillons pré-fuite
            n_fuite_critique: Nombre d'échantillons fuite critique
            ajouter_timestamp: Ajouter une colonne timestamp
        
        Returns:
            DataFrame pandas avec toutes les données
        """
        
        print("=" * 70)
        print("   GÉNÉRATION DU DATASET DE FUITES D'AIR COMPRIMÉ")
        print("=" * 70)
        
        print(f"\n📊 Configuration:")
        print(f"   • Données normales (état 0):          {n_normal:,}")
        print(f"   • Données pré-fuite (état 1):         {n_pre_fuite:,}")
        print(f"   • Données fuite critique (état 2):    {n_fuite_critique:,}")
        print(f"   • Total:                               {n_normal + n_pre_fuite + n_fuite_critique:,}")
        
        donnees = []
        timestamps = []
        date_debut = datetime.now() - timedelta(days=30)
        
        print(f"\n🔧 Génération en cours...")
        
        # Générer données normales (état 0)
        print(f"   Génération état 0 (Normal)...", end=" ")
        for i in range(n_normal):
            lecture = self.generer_lecture_capteur(etat=0)
            donnees.append(lecture)
            if ajouter_timestamp:
                timestamps.append(date_debut + timedelta(minutes=i*5))
        print("✓")
        
        # Générer données pré-fuite (état 1)
        print(f"   Génération état 1 (Pré-fuite)...", end=" ")
        for i in range(n_pre_fuite):
            lecture = self.generer_lecture_capteur(etat=1)
            donnees.append(lecture)
            if ajouter_timestamp:
                timestamps.append(date_debut + timedelta(minutes=(n_normal + i)*5))
        print("✓")
        
        # Générer données fuite critique (état 2)
        print(f"   Génération état 2 (Fuite critique)...", end=" ")
        for i in range(n_fuite_critique):
            lecture = self.generer_lecture_capteur(etat=2)
            donnees.append(lecture)
            if ajouter_timestamp:
                timestamps.append(date_debut + timedelta(minutes=(n_normal + n_pre_fuite + i)*5))
        print("✓")
        
        # Créer le DataFrame
        df = pd.DataFrame(donnees)
        
        if ajouter_timestamp:
            df['timestamp'] = timestamps
            # Réorganiser les colonnes
            colonnes = ['timestamp', 'vibration_hz', 'pression_bar', 'debit_m3h', 
                       'temperature_c', 'ultrasons_db', 'etat']
            df = df[colonnes]
        
        # Mélanger les données
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"\n✅ Dataset généré avec succès!")
        print(f"\n📈 Statistiques:")
        print(f"   • Total d'échantillons: {len(df):,}")
        print(f"   • Colonnes: {len(df.columns)}")
        print(f"\n📊 Distribution des états:")
        distribution = df['etat'].value_counts().sort_index()
        for etat, count in distribution.items():
            pourcentage = (count / len(df)) * 100
            label = {0: 'Normal', 1: 'Pré-fuite', 2: 'Fuite critique'}[etat]
            print(f"   • État {etat} ({label}): {count:,} ({pourcentage:.1f}%)")
        
        print(f"\n📋 Aperçu des données (premières lignes):")
        print(df.head(10).to_string(index=False))
        
        print(f"\n📊 Statistiques descriptives par état:")
        for etat in [0, 1, 2]:
            label = {0: 'NORMAL', 1: 'PRÉ-FUITE', 2: 'FUITE CRITIQUE'}[etat]
            print(f"\n   {label} (État {etat}):")
            subset = df[df['etat'] == etat]
            stats = subset[['vibration_hz', 'pression_bar', 'debit_m3h', 
                          'temperature_c', 'ultrasons_db']].describe().loc[['mean', 'std']]
            print(stats.to_string())
        
        return df
    
    def sauvegarder_dataset(self, df, nom_fichier='dataset_fuites.csv'):
        """
        Sauvegarde le dataset en CSV
        
        Args:
            df: DataFrame à sauvegarder
            nom_fichier: Nom du fichier de sortie
        """
        df.to_csv(nom_fichier, index=False)
        print(f"\n💾 Dataset sauvegardé: {nom_fichier}")
        print(f"   Taille du fichier: {len(df):,} lignes × {len(df.columns)} colonnes")
        print(f"\n🎯 Le dataset est prêt pour l'entraînement d'un modèle IA!")


def main():
    """Fonction principale pour générer le dataset"""
    
    # Créer le générateur
    generateur = GenerateurDatasetFuites(random_state=42)
    
    # Générer le dataset
    # Vous pouvez modifier ces nombres selon vos besoins
    df = generateur.generer_dataset(
        n_normal=3000,           # 50% de données normales
        n_pre_fuite=1500,        # 25% de pré-fuites
        n_fuite_critique=1500,   # 25% de fuites critiques
        ajouter_timestamp=True   # Inclure les timestamps
    )
    
    # Sauvegarder le dataset
    generateur.sauvegarder_dataset(df, 'dataset_fuites_air_comprime.csv')
    
    print("\n" + "=" * 70)
    print("✅ GÉNÉRATION TERMINÉE!")
    print("=" * 70)
    print("\n📝 Prochaines étapes:")
    print("   1. Ouvrir 'dataset_fuites_air_comprime.csv' dans Excel/Python")
    print("   2. Analyser les données")
    print("   3. Entraîner votre modèle IA avec ce dataset")
    print("\n💡 Colonnes du dataset:")
    print("   • timestamp: Date et heure de la mesure")
    print("   • vibration_hz: Vibration en Hz")
    print("   • pression_bar: Pression en bar")
    print("   • debit_m3h: Débit en m³/h")
    print("   • temperature_c: Température en °C")
    print("   • ultrasons_db: Niveau ultrasons en dB")
    print("   • etat: 0=Normal | 1=Pré-fuite | 2=Fuite critique")
    
    return df


if __name__ == "__main__":
    # Générer le dataset
    dataset = main()
    
    # Exemple d'utilisation: afficher quelques statistiques
    print("\n" + "=" * 70)
    print("   EXEMPLES DE DONNÉES PAR ÉTAT")
    print("=" * 70)
    
    for etat in [0, 1, 2]:
        label = {0: '✅ NORMAL', 1: '⚠️ PRÉ-FUITE', 2: '🚨 FUITE CRITIQUE'}[etat]
        print(f"\n{label}:")
        echantillon = dataset[dataset['etat'] == etat].head(3)
        print(echantillon[['vibration_hz', 'pression_bar', 'debit_m3h', 
                          'temperature_c', 'ultrasons_db', 'etat']].to_string(index=False))
