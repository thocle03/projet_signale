"""
Programme pour le telechargement et traitement via fft de fichiers audios
"""

import os
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import requests
import matplotlib.pyplot as plt


# === PARTIE 1 : Téléchargement des fichiers audio ===


def create_folder(folder_path):
    """
    Crée un dossier s'il n'existe pas.

    Paramètres :
    - folder_path (str) : Chemin du dossier à créer.
    """
    Path(folder_path).mkdir(parents=True, exist_ok=True)
    print(f"Folder '{folder_path}' is ready.")


def download_file(file_url, local_file_path):
    """
    Télécharge un fichier depuis une URL et l'enregistre localement.

    Paramètres :
    - file_url (str) : URL du fichier à télécharger.
    - local_file_path (Path) : Chemin complet pour sauvegarder le fichier localement.

    Exceptions :
    - Lève une exception si le téléchargement échoue.
    """
    try:
        print(f"Downloading: {local_file_path.name}")
        response = requests.get(file_url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(local_file_path, "wb") as f:
                f.write(response.content)
            print(f"Successfully downloaded: {local_file_path.name}")
        else:
            raise ValueError(
                f"Failed to download {local_file_path.name}."
                f"HTTP status code: {response.status_code}"
            )
    except Exception as e:
        print(f"Error downloading {local_file_path.name}: {e}")


def download_audio_files(excel_path, raw_folder, num_rows):
    """
    Télécharge les fichiers audio listés dans un fichier Excel.

    Paramètres :
    - excel_path (str) : Chemin du fichier Excel contenant les informations des fichiers audio.
    - raw_folder (str) : Dossier où les fichiers audio bruts seront sauvegardés.
    - num_rows (int) : Nombre de lignes à traiter dans le fichier Excel.
    """
    # Charger le fichier Excel
    df = pd.read_excel(excel_path)
    if num_rows > 0:
        df = df.head(num_rows)  # Ne garder que les x premières lignes
    create_folder(raw_folder)
    error_log = []

    for _, row in df.iterrows():
        file_url = row["file_url"]
        file_name = row["file_name"]
        local_file_path = Path(raw_folder) / file_name

        if (Path(raw_folder) / file_name).exists():
            print(f"File already exists, skipping download: {file_name}")
            continue

        try:
            download_file(file_url, local_file_path)
        except RuntimeError as e:
            print(e)  # Log the error
            error_log.append(str(e))  # Ajouter l'erreur dans la liste

    # Sauvegarder les erreurs dans un fichier
    if error_log:
        with open("errors.txt", "w", encoding="utf-8") as error_file:
            for error in error_log:
                error_file.write(error + "\n")
        print("Error log saved to 'errors.txt'.")


# === PARTIE 2 : Traitement des fichiers audio ===


def remove_background_noise(
    rbn_raw_folder, rbn_processed_folder, min_frequency=5000, max_frequency=10000
):
    """
    Traite les fichiers audio pour supprimer le bruit de fond à l'aide de FFT, en bloquant les fréquences
    en dehors de la plage [min_frequency, max_frequency].

    Paramètres :
    - rbn_raw_folder (str) : Dossier contenant les fichiers audio bruts.
    - rbn_processed_folder (str) : Dossier où les fichiers traités seront sauvegardés.
    - min_frequency (int) : Fréquence minimale à conserver (en Hz).
    - max_frequency (int) : Fréquence maximale à conserver (en Hz).
    """
    create_folder(rbn_processed_folder)

    for file_name in os.listdir(rbn_raw_folder):
        if file_name.endswith(".mp3") or file_name.endswith(".wav"):
            try:
                file_path = os.path.join(rbn_raw_folder, file_name)

                # Charger l'audio
                signal, sr = librosa.load(file_path, sr=None)

                # Appliquer la FFT
                fft_signal = np.fft.fft(signal)

                # Calculer les fréquences correspondantes aux bins FFT
                freqs = np.fft.fftfreq(len(fft_signal), 1 / sr)

                # Créer un masque pour garder uniquement les fréquences entre min_frequency et max_frequency
                mask = (np.abs(freqs) > min_frequency) & (np.abs(freqs) < max_frequency)

                # Appliquer le masque
                filtered_fft_signal = fft_signal * mask

                # Retour au domaine temporel via l'IFFT
                filtered_signal = np.fft.ifft(filtered_fft_signal).real

                # Sauvegarder le fichier traité
                output_path = os.path.join(rbn_processed_folder, file_name)
                sf.write(output_path, filtered_signal, sr)
                print(f"Processed and saved: {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")


if __name__ == "__main__":

    EXCEL_PATH = "data.xlsx"
    RAW_FOLDER = "raw_audios2"
    PROCESSED_FOLDER = "processed_audios2"
    NUM_ROWS = 5
    # NOISE_THRESHOLD = 10

    # Étape 1 : Télécharger les fichiers audio
    download_audio_files(EXCEL_PATH, RAW_FOLDER, NUM_ROWS)

    # Étape 2 : Traiter les fichiers audio
    remove_background_noise(RAW_FOLDER, PROCESSED_FOLDER)
