import os
import requests
import pandas as pd
import numpy as np
import librosa
import scipy.fftpack
from tensorflow.keras import layers, models
from openpyxl import Workbook
from pathlib import Path

# Paramètres de chemin de fichier
input_excel = "data.xlsx"  # Nom exact du fichier d'entrée
output_folder = "dossier_audio_traite"  # Dossier de sortie pour les fichiers audio traités
excel_output = "classification_audio.xlsx"  # Nom du fichier Excel de sortie
target_duration = 2.0  # Durée cible en secondes pour chaque fichier audio

# Créer le dossier de sortie s'il n'existe pas
Path(output_folder).mkdir(parents=True, exist_ok=True)

# Charger les métadonnées du fichier Excel
metadata_df = pd.read_excel(input_excel)

# Fonction pour télécharger un fichier audio
def download_audio(file_url, save_path):
    try:
        response = requests.get(file_url, stream=True)
        if response.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(response.content)
            print(f"Téléchargé : {save_path}")
        else:
            print(f"Erreur lors du téléchargement de {file_url} (code {response.status_code})")
            return False
    except Exception as e:
        print(f"Erreur lors du téléchargement de {file_url}: {e}")
        return False
    return True

# Fonction pour traiter un fichier audio avec la FFT et normaliser sa longueur
def process_audio_with_fft(audio_path, sr=22050, target_duration=2.0):
    try:
        y, _ = librosa.load(audio_path, sr=sr, duration=target_duration)
        if len(y) < sr * target_duration:
            # Compléter l'audio pour qu'il ait exactement la longueur cible
            y = np.pad(y, (0, int(sr * target_duration) - len(y)), mode='constant')
        elif len(y) > sr * target_duration:
            # Tronquer l'audio pour qu'il ait exactement la longueur cible
            y = y[:int(sr * target_duration)]
        
        fft_values = scipy.fftpack.fft(y)
        frequencies = scipy.fftpack.fftfreq(len(y), d=1/sr)
        return np.abs(fft_values[:len(frequencies)//2])
    except Exception as e:
        print(f"Erreur lors du traitement du fichier {audio_path}: {e}")
        return None

# Modèle CNN pour la classification
def create_cnn_model(input_shape):
    model = models.Sequential([
        layers.Conv1D(16, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling1D(2),
        layers.Conv1D(32, 3, activation='relu'),
        layers.MaxPooling1D(2),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')  # Modifier le nombre de classes si nécessaire
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Définir input_shape basé sur la durée cible
input_shape = (int(target_duration * 22050 // 2), 1)  # On divise par 2 car on prend la moitié des valeurs FFT

# Charger ou entraîner le modèle CNN
model = create_cnn_model(input_shape=input_shape)

# Liste des catégories (par exemple, selon le type ou l'espèce)
categories = ['call', 'song', 'alarm', 'other']  # Ajuster les classes selon les besoins

# Liste pour stocker les résultats
results = []

# Traitement de chaque fichier audio
for index, row in metadata_df.iterrows():
    file_id = row['file_id']
    file_url = row['file_url']
    file_name = f"{file_id}.wav"  # Nom de fichier local
    save_path = os.path.join(output_folder, file_name)
    
    # Télécharger le fichier audio
    if not download_audio(file_url, save_path):
        print(f"Erreur: Impossible de télécharger le fichier audio avec l'ID {file_id}. Passage au fichier suivant.")
        continue  # Passer au fichier suivant en cas d'erreur de téléchargement
    
    # Appliquer la FFT
    fft_features = process_audio_with_fft(save_path)
    if fft_features is None:
        print(f"Erreur: Impossible de traiter le fichier audio {save_path}. Passage au fichier suivant.")
        continue  # Passer au fichier suivant en cas d'erreur de traitement

    # Redimensionner pour le CNN
    fft_features = fft_features.reshape(-1, 1)

    # Classification avec le modèle CNN
    prediction = model.predict(np.array([fft_features]))
    predicted_class = categories[np.argmax(prediction)]

    # Enregistrer les résultats
    results.append({
        'file_id': file_id,
        'file_name': file_name,
        'predicted_class': predicted_class,
        **row.to_dict()  # Ajouter toutes les colonnes de métadonnées originales
    })

# Créer un fichier Excel avec les résultats de classification
results_df = pd.DataFrame(results)
results_df.to_excel(excel_output, index=False)

print(f"Traitement terminé. Les fichiers audio traités sont enregistrés dans {output_folder} et les résultats dans {excel_output}.")
