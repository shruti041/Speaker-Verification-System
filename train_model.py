import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import joblib
from audio import extract_mfcc  

def load_dataset(base_dir, target_speaker):
    X, y = [], []
    for speaker in os.listdir(base_dir):
        speaker_path = os.path.join(base_dir, speaker)
        if not os.path.isdir(speaker_path):
            continue

        label = 1 if speaker == target_speaker else 0

        for file in os.listdir(speaker_path):
            if file.endswith(".wav"):
                file_path = os.path.join(speaker_path, file)
                features = extract_mfcc(file_path)
                if features.shape[0] == 13:
                    X.append(features)
                    y.append(label)

    X, y = np.array(X), np.array(y)
    print(f"Loaded {len(X)} samples.")
    print(f"Class distribution: {np.bincount(y)}")

    if len(np.unique(y)) < 2:
        raise ValueError("Need at least two classes in the dataset (target and non-target).")

    return X, y

DATASET_DIR = "dataset"
TARGET_SPEAKER = "Nelson_Mandela"

X, y = load_dataset(DATASET_DIR, TARGET_SPEAKER)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = make_pipeline(StandardScaler(), SVC(probability=True))
model.fit(X_train, y_train)

joblib.dump({'model': model, 'target_speaker': TARGET_SPEAKER}, "svm_speaker_verification.pkl")

# Evaluate using threshold on probabilities
y_prob = model.predict_proba(X_test)[:, 1]
threshold = 0.5
y_pred = (y_prob >= threshold).astype(int)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
