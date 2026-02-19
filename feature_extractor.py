import librosa
import numpy as np

def extract_features_with_stats(file_path, n_mfcc=13):
    y, sr = librosa.load(file_path, sr=16000)
    y, _ = librosa.effects.trim(y, top_db=25)

    # ===== MFCCs =====
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    mfcc_stack = np.vstack([mfcc, delta, delta2])
    mfcc_mean = np.mean(mfcc_stack, axis=1)
    mfcc_std = np.std(mfcc_stack, axis=1)

    # ===== Pitch =====
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    pitch = pitches[mags > np.median(mags)]
    pitch_mean = float(np.mean(pitch)) if pitch.size else 0.0
    pitch_std = float(np.std(pitch)) if pitch.size else 0.0

    # ===== Jitter =====
    jitter = float(np.mean(np.abs(np.diff(pitch)))) if pitch.size > 1 else 0.0

    # ===== Shimmer =====
    rms = librosa.feature.rms(y=y)[0]
    shimmer = float(np.mean(np.abs(np.diff(rms)))) if rms.size > 1 else 0.0

    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [pitch_mean, pitch_std, jitter, shimmer]
    ])

    stats = {
        "pitch_mean": pitch_mean,
        "pitch_std": pitch_std,
        "jitter": jitter,
        "shimmer": shimmer,
        "mfcc_std_mean": float(np.mean(mfcc_std))
    }

    return features, stats


def generate_dynamic_explanation(stats, confidence):
    reasons = []

    if stats["pitch_std"] < 15:
        reasons.append("very stable pitch")
    if stats["jitter"] < 0.002:
        reasons.append("low micro-variations in pitch")
    if stats["shimmer"] < 0.005:
        reasons.append("unnaturally consistent loudness")
    if stats["mfcc_std_mean"] < 8:
        reasons.append("limited spectral variation")

    if not reasons:
        reasons.append("natural vocal variability")

    explanation = (
        f"Decision based on {', '.join(reasons)}. "
        f"Model confidence is {confidence:.2f}."
    )

    return explanation
