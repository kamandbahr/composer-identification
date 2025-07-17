import os
import glob
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import pywt
import soundfile as sf

# treshold of energy was asked and tuned by AI
INPUT_DIR = '/Users/kamand/GiantMIDI_Workspace/mp3s_piano_solo/converted_wavs'
OUTPUT_DIR = '/Users/kamand/preprocessing'
SEGMENT_LENGTH = 30  # seconds
ENERGY_THRESHOLD = 0.01  # Tune as needed

# managing my files here I can have every feature seperately
# the folders will be created by it's function if it doesn't exist
FEATURE_FOLDERS = {
    'mel_png': 'mel_spectrograms/png',
    'mel_npy': 'mel_spectrograms/npy',
    'wavelet': 'wavelet_features',
    'mfcc': 'mfcc_features',
    'chroma': 'chroma_features',
    'spectral_contrast': 'spectral_contrast_features',
    'tempo': 'tempo_features',
    'zero_crossing': 'zero_crossing_rate_features',
    'energy': 'energy_features',
}

def create_folders():
    for folder in FEATURE_FOLDERS.values():
        path = os.path.join(OUTPUT_DIR, folder)
        os.makedirs(path, exist_ok=True)

# energy
def compute_energy(y):
    return np.mean(y ** 2)

# mel spectogram
def save_mel_spectrogram(y, sr, out_base):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Save numpy array
    np.save(f'{out_base}.npy', S_dB)
    # Save PNG
    plt.figure(figsize=(6, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.savefig(f'{out_base}.png', bbox_inches='tight', pad_inches=0)
    plt.close()

# save wavelet transform in the folder that is been created
# we'll use the mean and std of the coefficients for each level as features
def save_wavelet_features(y, out_path):
    coeffs = pywt.wavedec(y, 'db4', level=4)
    features = np.concatenate([np.array([np.mean(c), np.std(c)]) for c in coeffs])
    np.save(out_path, features)

# save MFCCs
def save_mfcc(y, sr, out_path):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    np.save(out_path, mfcc)

# chroma features
def save_chroma(y, sr, out_path):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    np.save(out_path, chroma)

# spectral contrast
def save_spectral_contrast(y, sr, out_path):
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    np.save(out_path, contrast)

# tempo/beat features
def save_tempo(y, sr, out_path):
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    np.savez(out_path, tempo=tempo, n_beats=len(beats), beats=beats)

# zero-crossing rate
def save_zero_crossing(y, out_path):
    zcr = librosa.feature.zero_crossing_rate(y)
    np.save(out_path, zcr)

# save energy
def save_energy(y, out_path):
    energy = compute_energy(y)
    np.save(out_path, np.array([energy]))

# Main processing function
def process_audio_files():
    create_folders()
    flac_files = glob.glob(os.path.join(INPUT_DIR, '*.flac'))
    total_files = len(flac_files)
    for file_idx, flac_path in enumerate(flac_files):
        song_name = os.path.splitext(os.path.basename(flac_path))[0]
        y, sr = sf.read(flac_path)
        if y.ndim > 1:
            y = np.mean(y, axis=1)  # Convert to mono if stereo
        total_duration = len(y) / sr
        num_segments = int(np.ceil(total_duration / SEGMENT_LENGTH))
        for i in range(num_segments):
            start_sample = int(i * SEGMENT_LENGTH * sr)
            end_sample = int(min((i + 1) * SEGMENT_LENGTH * sr, len(y)))
            segment = y[start_sample:end_sample]
            if len(segment) < 1:
                continue
            energy = compute_energy(segment)
            if energy < ENERGY_THRESHOLD:
                continue  # Skip low-energy segments like quiet segments will be discussed in the presentation
            seq_name = f'{song_name}_seq{i+1}'
            # check if this segment has already been processed (mel npy file exists)
            # the part is been added because of the kernel stops in the jupyter notebook 
            mel_npy_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['mel_npy'], seq_name + '.npy')
            if os.path.exists(mel_npy_path):
                continue  # skip already processed segment
            # mel spectrogram
            mel_base_png = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['mel_png'], seq_name)
            mel_base_npy = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['mel_npy'], seq_name)
            save_mel_spectrogram(segment, sr, mel_base_npy)  # Save numpy
            save_mel_spectrogram(segment, sr, mel_base_png)  # Save PNG
            # wavelet
            wavelet_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['wavelet'], seq_name + '.npy')
            save_wavelet_features(segment, wavelet_path)
            # MFCC and chroma
            mfcc_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['mfcc'], seq_name + '.npy')
            save_mfcc(segment, sr, mfcc_path)
            chroma_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['chroma'], seq_name + '.npy')
            save_chroma(segment, sr, chroma_path)

            # spectral contrast
            contrast_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['spectral_contrast'], seq_name + '.npy')
            save_spectral_contrast(segment, sr, contrast_path)
            # tempo
            tempo_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['tempo'], seq_name + '.npy')
            save_tempo(segment, sr, tempo_path)
            # zero-crossing rate
            zcr_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['zero_crossing'], seq_name + '.npy')
            save_zero_crossing(segment, zcr_path)
            # energy
            energy_path = os.path.join(OUTPUT_DIR, FEATURE_FOLDERS['energy'], seq_name + '.npy')
            save_energy(segment, energy_path)
        # progress after each file
        percent_done = ((file_idx + 1) / total_files) * 100
        print(f"Processed {file_idx + 1}/{total_files} files ({percent_done:.2f}%)")

if __name__ == '__main__':
    process_audio_files() 
