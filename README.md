# 🎙️ Speaker Verification System
## 📌 Project Overview
This project implements a real-time speaker verification system using machine learning. The system listens to microphone input and classifies whether the incoming speaker is the target speaker or not, based on audio features extracted from sample .wav files.
## 🔍 Objectives
**1.** Pre-process audio and extract meaningful features (MFCCs). <br>
**2.** Train a binary classifier to distinguish the target speaker from others.<br>
**3.** Evaluate the model using relevant metrics like accuracy and F1-score.<br>
**4.** Deploy the solution with a real-time Streamlit web interface.
## 🗂️ Dataset
-**Source:** Kaggle - Speaker Recognition Dataset
-**Format:** Directory of .wav files, grouped by speaker names (each folder = one speaker)
-**Sampling Rate:** Standardized to 16,000 Hz
