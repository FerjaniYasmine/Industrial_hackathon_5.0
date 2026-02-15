"""
API FLASK - Intégration du modèle IA avec le Digital Twin
===========================================================
Expose votre modèle d'IA en API REST pour le Digital Twin
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
from datetime import datetime
import sys
import os

# Importer votre modèle
from modele_ia_fuites import ModeleDetectionFuites

# ========== INITIALISATION ==========
app = Flask(__name__)
CORS(app)

print("\n" + "="*70)
print("   API MODÈLE IA - DÉTECTION DE FUITES")
print("="*70)
print("\n🚀 Démarrage de l'API...")

# Charger le modèle
try:
    modele = ModeleDetectionFuites.charger('modele_ia_fuites.pkl')
    print("✅ Modèle IA chargé avec succès")
    API_READY = True
except FileNotFoundError:
    print("❌ Modèle non trouvé! Veuillez exécuter: python modele_ia_fuites.py")
    API_READY = False
    modele = None

# ========== ENDPOINTS API ==========

@app.route('/health', methods=['GET'])
def health():
    """Vérifier si l'API est actif"""
    return jsonify({
        'status': 'OK',
        'model_loaded': API_READY,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prédire le risque de fuite pour UN tuyau
    
    Payload JSON:
    {
        "pipe_id": "A-12",
        "pressure": 7.0,
        "vibration": 25,
        "temperature": 60,
        "flowRate": 150,
        "acoustic": 45
    }
    """
    
    if not API_READY:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        # Mapper les données du Digital Twin au format du modèle
        sensor_data = {
            'pression_bar': float(data.get('pressure', 7.0)),
            'vibration_hz': float(data.get('vibration', 25)),
            'temperature_c': float(data.get('temperature', 60)),
            'debit_m3h': float(data.get('flowRate', 150)),
            'ultrasons_db': float(data.get('acoustic', 45))
        }
        
        # Prédiction
        prediction = modele.predire(sensor_data)
        
        # Formater la réponse
        response = {
            'success': True,
            'pipe_id': data.get('pipe_id', 'unknown'),
            'prediction': {
                'leak_state': prediction['etat'],
                'leak_state_label': prediction['label'],
                'risk_level': prediction['niveau_risque'],
                'leak_probability': prediction['probabilites']['fuite_critique'],
                'confidence': prediction['confiance'],
                'recommendation': prediction['recommandation'],
                
                'probabilities': {
                    'normal': prediction['probabilites']['normal'],
                    'pre_fuite': prediction['probabilites']['pre_fuite'],
                    'fuite_critique': prediction['probabilites']['fuite_critique']
                },
                
                'sensor_analysis': prediction['analyse_capteurs'],
                
                'timestamp': datetime.now().isoformat(),
                'model_version': 'modele_ia_fuites.pkl'
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"❌ Erreur: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """
    Prédire pour PLUSIEURS tuyaux à la fois
    """
    
    if not API_READY:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        pipes_data = request.get_json()
        predictions = {}
        critical_count = 0
        warning_count = 0
        
        for pipe_id, sensor_data in pipes_data.items():
            # Mapper les données
            mapped_data = {
                'pression_bar': float(sensor_data.get('pressure', 7.0)),
                'vibration_hz': float(sensor_data.get('vibration', 25)),
                'temperature_c': float(sensor_data.get('temperature', 60)),
                'debit_m3h': float(sensor_data.get('flowRate', 150)),
                'ultrasons_db': float(sensor_data.get('acoustic', 45))
            }
            
            # Prédiction
            prediction = modele.predire(mapped_data)
            
            if prediction['etat'] == 2:
                critical_count += 1
            elif prediction['etat'] == 1:
                warning_count += 1
            
            predictions[pipe_id] = {
                'leak_state': prediction['etat'],
                'leak_state_label': prediction['label'],
                'risk_level': prediction['niveau_risque'],
                'leak_probability': prediction['probabilites']['fuite_critique'],
                'confidence': prediction['confiance'],
                'probabilities': prediction['probabilites'],
                'sensor_analysis': prediction['analyse_capteurs'],
                'recommendation': prediction['recommandation'],
                'timestamp': datetime.now().isoformat()
            }
        
        return jsonify({
            'success': True,
            'total_pipes': len(predictions),
            'critical_pipes': critical_count,
            'warning_pipes': warning_count,
            'healthy_pipes': len(predictions) - critical_count - warning_count,
            'predictions': predictions,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        print(f"❌ Erreur batch: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/model-info', methods=['GET'])
def model_info():
    """Informations sur le modèle"""
    
    if not API_READY:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': modele.type_modele,
        'features': modele.noms_features,
        'states': modele.labels_etats,
        'accuracy': modele.historique_entrainement.get('precision_globale', 0),
        'training_date': modele.historique_entrainement.get('date', 'unknown'),
        'feature_importance': modele.historique_entrainement.get('importance_features', [])
    })

if __name__ == '__main__':
    print(f"\n🚀 Serveur démarré: http://localhost:5000")
    print(f"\n📊 Endpoints disponibles:")
    print(f"   GET  /health              - Vérifier statut")
    print(f"   POST /predict             - Prédiction simple")
    print(f"   POST /predict-batch       - Prédictions batch")
    print(f"   GET  /model-info          - Infos du modèle")
    print(f"\n" + "="*70)
    print(f"Appuyez sur Ctrl+C pour arrêter\n")
    
    app.run(debug=True, port=5000, threaded=True)