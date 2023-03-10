from flask import Flask, jsonify, request
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
import numpy as np
import pandas as pd
import pickle
import shap

app = Flask(__name__)

# fonction pour charger une table enregistrée en pickle
def load_pickle(filename):
    with open(f'./{filename}.pickle', 'rb') as f:
        return pickle.load(f)

#df = load_pickle('X_test_sample')
# Préparation des données pour les besoins de l'application

# Récupération de la liste client
#df = df.set_index('SK_ID_CURR')

df = load_pickle('data_filtered')
df = df.set_index('SK_ID_CURR')
df_cli = df[[c for c in df.columns if c not in ['TARGET']]]
list_clients = list(df_cli.index)

pipeline = load_pickle('model')
scaler = StandardScaler()
scaler_df = pd.DataFrame(scaler.fit_transform(df_cli), 
                        columns=df_cli.columns,
                        index=df_cli.index)
model = pipeline.steps[1][1]

threshold = 0.7

# Récupérer la liste des clients
@app.route('/clients/',  methods=['GET'])
def clients():
    response = jsonify(list_clients)
    return response

# Récupération des données du clients
dict_ = df_cli.to_dict(orient='index')

# Pour tous les clients
@app.route('/client/')
def get_my_data():
    # Renvoyer le dictionnaire en réponse
    return jsonify(dict_)

# Pour un client selectionné
@app.route('/client/<id>',  methods=['GET'])
def client(id):
    data_client = df_cli[df_cli.index == int(id)]
    response = data_client.to_dict(orient='index')
    return jsonify(response)

# Pour la prédiction du client (avec la probabilité d'obtenir le prêt)
@app.route('/predict/<id>',  methods=['GET'])
def predict(id):
    data_cli = scaler_df[scaler_df.index==int(id)]
    y_prob = model.predict_proba(data_cli)[:, 1][0]
    y_predict = int((y_prob > threshold) * 1)
    response = {"id_": id, 
        "threshold": threshold, 
        "predict": y_predict,
        "probability": y_prob}
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
