import librosa
import numpy as np

def extract_mfcc(file_path, n_mfcc=13, normalize=True):
    try:
        y, sr = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if normalize:
            mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

        mfcc_mean = np.mean(mfcc.T, axis=0)
        return mfcc_mean
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return np.zeros(n_mfcc)
