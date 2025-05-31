import streamlit as st
import numpy as np
import joblib
import librosa
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase
import av
import queue

@st.cache_resource
def load_model():
    return joblib.load("svm_speaker_verification.pkl")

model_bundle = load_model()
model = model_bundle['model']
target_speaker = model_bundle['target_speaker']

pred_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        print("recv called")
        audio = frame.to_ndarray(format="flt32").flatten()
        print(f"Audio samples length: {len(audio)}")

        # Minimum length check
        if len(audio) < 4000:
            return frame

        try:
            sample_rate = frame.sample_rate or 48000  # fallback if None
            audio_resampled = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

            # Check for silence
            if np.all(audio_resampled == 0):
                print("Silent audio detected.")
                return frame

            mfcc = librosa.feature.mfcc(y=audio_resampled, sr=16000, n_mfcc=13)
            mfcc_mean = np.mean(mfcc.T, axis=0).reshape(1, -1)
            print(f"MFCC shape: {mfcc_mean.shape}")
        except Exception as e:
            print("Feature extraction error:", e)
            return frame

        try:
            prob = model.predict_proba(mfcc_mean)[0][1]
            label = "🎯 Target Speaker" if prob >= 0.5 else "❌ Non-Target Speaker"
            print(f"Prediction: {label} with prob {prob:.2f}")
            if pred_queue.qsize() < 5:
                pred_queue.put((label, prob))
        except Exception as e:
            print("Prediction error:", e)

        return frame


st.title(f"🎙️ Real-Time Speaker Verification")
st.subheader(f"Target Speaker: `{target_speaker}`")

webrtc_ctx = webrtc_streamer(
    key="verifier",
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True,
)

placeholder = st.empty()

# Periodically check for prediction
if webrtc_ctx.state.playing:
    while True:
        try:
            label, prob = pred_queue.get(timeout=1)
            placeholder.markdown(f"### Prediction: **{label}**")
            placeholder.write(f"Confidence Score: `{prob:.2f}`")
        except queue.Empty:
            break
else:
    st.info("Click **Start** above to begin microphone streaming.")
