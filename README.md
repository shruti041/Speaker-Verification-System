# 🎙️ Speaker Verification System
## 📌 Project Overview
This project implements a real-time speaker verification system using machine learning. The system listens to microphone input and classifies whether the incoming speaker is the target speaker or not, based on audio features extracted from sample .wav files.
## 🔍 Objectives
**1.** Pre-process audio and extract meaningful features (MFCCs). <br>
**2.** Train a binary classifier to distinguish the target speaker from others.<br>
**3.** Evaluate the model using relevant metrics like accuracy and F1-score.<br>
**4.** Deploy the solution with a real-time Streamlit web interface.
## 🗂️ Dataset
-**Source:** Kaggle - Speaker Recognition Dataset <br>
-**Format:** Directory of .wav files, grouped by speaker names (each folder = one speaker)<br>
-**Sampling Rate:** Standardized to 16,000 Hz
## 🎵 MFCC Feature Extraction
Mel-Frequency Cepstral Coefficients (MFCCs) capture the timbral aspects of speech:<br>
- Extracted using librosa.feature.mfcc<br>
- Each frame results in 13 MFCCs → take the mean across time to form a fixed-length vector<br>
- Normalization helps in robust model performance
## 🧠 Binary Classification for Speaker Verification
- This is a supervised learning problem<br>
- **Label 1:** Target speaker (e.g., "Nelson_Mandela")<br>
- **Label 0:** Non-target speakers<br>
- **Model:** Pipeline(StandardScaler + SVC(probability=True))
## 📈 Evaluation Metrics
- **Accuracy:** (TP + TN) / Total<br>
- **F1 Score:** Harmonic mean of precision and recall, better for class imbalance<br>
- **Threshold:** Based on probability score; default is 0.5 
## 🚀 Real-Time Speaker Verification App
- Use streamlit with streamlit-webrtc for live audio streaming.<br>
- Capture audio from microphone.<br>
- Extract MFCC features from live audio chunks.<br>
- Predict speaker identity probabilities using trained model.<br>
- Display real-time predictions and confidence scores. 
