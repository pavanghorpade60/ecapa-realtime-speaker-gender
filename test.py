import torch
import sounddevice as sd
import numpy as np
from model import ECAPA_gender


# -------------------------------------------------
# 1️⃣ Load Model
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECAPA_gender.from_pretrained("JaesungHuh/voice-gender-classifier")
model.to(device)
model.eval()


# -------------------------------------------------
# 2️⃣ Record Audio
# -------------------------------------------------
def record_audio(duration: int = 5, fs: int = 16000) -> torch.Tensor:
    print("🎤 Speak now...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    print("✅ Recording finished.")

    audio = np.squeeze(audio)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)

    return audio_tensor


# -------------------------------------------------
# 3️⃣ Silence Detection (STRONGER)
# -------------------------------------------------
def is_silence(audio_tensor, silence_threshold=0.01):
    rms = torch.sqrt(torch.mean(audio_tensor ** 2))
    avg_energy = torch.mean(torch.abs(audio_tensor))

    print(f"Raw RMS Energy: {rms.item():.6f}")
    print(f"Raw Avg Energy: {avg_energy.item():.6f}")

    # If both are low → silence
    return rms.item() < silence_threshold and avg_energy.item() < silence_threshold


# -------------------------------------------------
# 4️⃣ Normalize AFTER silence check
# -------------------------------------------------
def normalize_audio(audio_tensor):
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        audio_tensor = audio_tensor / max_val
    return audio_tensor


# -------------------------------------------------
# 5️⃣ Gender Prediction
# -------------------------------------------------
def predict_live(model, audio_tensor: torch.Tensor, device: torch.device) -> str:

    audio_tensor = audio_tensor.squeeze(0).float()

    # ---------------------------
    # Silence Check BEFORE normalization
    # ---------------------------
    if is_silence(audio_tensor):
        return "silence"

    # ---------------------------
    # Normalize only if speech detected
    # ---------------------------
    audio_tensor = normalize_audio(audio_tensor)

    # Segment processing
    segment_length = 32000
    segments = []

    for i in range(0, len(audio_tensor) - segment_length + 1, segment_length):
        segment = audio_tensor[i:i + segment_length]
        segments.append(segment.unsqueeze(0))

    if len(segments) == 0:
        segments.append(audio_tensor.unsqueeze(0))

    total_probs = torch.zeros(2).to(device)

    with torch.no_grad():
        for segment in segments:
            segment = segment.to(device)
            output = model(segment)
            probs = torch.softmax(output, dim=1)
            total_probs += probs.squeeze(0)

    avg_probs = total_probs / len(segments)

    male_score = avg_probs[0].item()
    female_score = avg_probs[1].item()

    confidence, prediction = torch.max(avg_probs, dim=0)
    difference = abs(male_score - female_score)

    print(f"Male probability   : {male_score:.4f}")
    print(f"Female probability : {female_score:.4f}")
    print(f"Confidence         : {confidence.item():.4f}")

    # Stronger confidence rule
    if confidence.item() < 0.75 or difference < 0.30:
        return "unknown"

    return "male" if prediction.item() == 0 else "female"


# -------------------------------------------------
# 6️⃣ Run
# -------------------------------------------------
if __name__ == "__main__":

    audio = record_audio(duration=6)

    result = predict_live(model, audio, device)

    print("\n🎤 Final Prediction:", result.upper())