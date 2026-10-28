# Vani Kavach — Acoustic Model Card & False Positive Investigation

## 1. Model Overview

- **Model Type**: 2D Convolutional Neural Network (CNN) on Log-Frequency Spectrograms
- **Architecture**: 3 Convolutional Blocks (Conv2D + MaxPool2D) + GlobalAveragePooling2D + Dense(64, ReLU) + Dropout(0.2) + Dense(1, Sigmoid)
- **Parameters**: 27,521 trainable parameters (~107.5 KB)
- **Framework**: Keras 3 with PyTorch Backend (`KERAS_BACKEND=torch`)
- **Weights Checkpoint**: `model/cnn_spectrogram_experimental.keras` (SHA256 verified)
- **Evaluation Status**: Experimental prototype. Evaluated only on a balanced 400-recording development subset of ASVspoof 2019 Logical Access (LA).

---

## 2. Label Mapping & Output Interpretation

The model output was verified against the training data, development samples (`vani_kavach_dev_samples.npz`), and evaluation metrics (`model/cnn_development_results.json`):

| Class Index | Label | Output Probability Range | Decision Threshold Rule |
| :--- | :--- | :--- | :--- |
| **0** | `bonafide` (genuine human voice) | $P(\text{spoof}) < 0.50$ | Classified as `bonafide` / `likely_genuine` |
| **1** | `spoof` (synthetic / converted voice) | $P(\text{spoof}) \ge 0.50$ | Classified as `spoof` / `potential_spoof` |

### Held-Out Evaluation on Balanced ASVspoof Development Samples ($N = 10$)

```text
Threshold:         0.50
Overall Accuracy:  10/10 (100.0%)
Confusion Matrix:  [[TN=6, FP=0], [FN=0, TP=4]]
Precision (Spoof): 1.0000
Recall (Spoof):    1.0000
F1 Score (Spoof):  1.0000
False Pos Rate:    0.0000
False Neg Rate:    0.0000
```

- **6 Bonafide Samples**: Predicted spoof probabilities between **0.0508** and **0.2971** $\rightarrow$ **100% accuracy** ($6/6$ correct).
- **4 Spoof Samples**: Predicted spoof probabilities between **0.9745** and **0.9994** $\rightarrow$ **100% accuracy** ($4/4$ correct).
- **Conclusion**: The label mapping in `app/detector.py` is **100% correct**. The false positive is **not** caused by label inversion.

---

## 3. False Positive Investigation (`Recording (2).wav`)

### Problem
The user's genuine natural voice recorded via laptop microphone (`Recording (2).wav`, 16.59s, 48 kHz stereo downsampled to 16 kHz mono) was classified by the model as:
- **Verdict**: `spoof` (`potential_spoof`)
- **Spoof Probability**: `0.9764` (originally `0.9929` without speech windowing)
- **Authenticity Score**: `0.0236`

### Root Cause Analysis

Empirical investigation revealed three compounding factors that cause this false positive:

#### A. Out-of-Distribution (OOD) Default Bias
The CNN was trained on only 2,000 samples from ASVspoof 2019 LA. All bonafide speech in ASVspoof comes from the VCTK corpus recorded with studio-grade condenser microphones in a hemi-anechoic chamber.
When tested on synthetic baseline inputs outside this studio distribution:
- **All-zero input (silence)**: Predicted spoof probability = **0.9979**
- **All-one input**: Predicted spoof probability = **0.9553**
- **Uniform random noise**: Predicted spoof probability = **0.9315**
The model has learned a narrow studio signature for bonafide speech. Anything that fails to match that specific profile defaults to **high-confidence spoof**.

#### B. Preprocessing Truncation & Silence Sensitivity (Mitigated by VAD)
- Previously, `extract_spectrogram` blindly truncated the time axis to the first 128 frames (1.28 seconds).
- `Recording (2).wav` has **0.77 seconds of leading ambient silence** before speech starts.
- Because `librosa.amplitude_to_db` normalizes relative to the peak of the entire 16.59s recording (`ref=np.max`), the leading 0.77s silence drops below $-80\text{ dB}$, producing **77 consecutive zero frames** (over 60% of the spectrogram).
- Feeding mostly flat silence directly triggered the model's all-zeros default spoof state (**0.9929**).
- **Mitigation Implemented**: Voice Activity Detection (VAD) and `select_best_speech_window()` now automatically extract the 1.28s window with the highest active speech energy, eliminating the leading silence artifact.

#### C. Acoustic Channel & Microphone Mismatch
When sliding 1.28s windows across the file are isolated to **active speech** segments:
- Speech window (1.28s - 2.56s): Spoof probability drops to **0.6920**.
- Speech window (8.32s - 9.60s): Spoof probability is **0.7447**.
While active speech lowers the spoof probability compared to silence (from 0.992 to 0.692), it remains above the 0.5 threshold because:
1. Consumer laptop microphone frequency response rolls off sharply below 150 Hz and introduces resonant peaks.
2. Room reverberation and background environmental noise alter the spectrogram texture.
3. The toy 27K-parameter CNN has zero acoustic data augmentation (no RIR convolution, no noise injection, no codec emulation).

---

## 4. Evaluation Results Summary

| Dataset / Sample | Sample Count | True Label | Pred Label | Avg Spoof Prob | Accuracy / Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ASVspoof Dev Bonafide** | 6 | `bonafide` | `bonafide` | 0.1354 | 100% Pass |
| **ASVspoof Dev Spoof** | 4 | `spoof` | `spoof` | 0.9914 | 100% Pass |
| **User Natural Voice (VAD windowed)** | 1 (16.59s) | `bonafide` | `spoof` | 0.9764 | **False Positive** |
| **Active Speech Slices** | 4 windows | `bonafide` | `spoof` | 0.7735 | **False Positive** |

Evaluation script: `evaluate_false_positives.py`
Structured results: `evaluation_results.json`

---

## 5. Trustworthiness Assessment

> [!WARNING]
> **Is the model currently trustworthy enough to present as a working spoof detector?**
>
> **NO.** The model is strictly an experimental prototype. While it achieves 93.25% on its closed-world ASVspoof development subset, it fails to generalize to real consumer microphone recordings due to severe channel mismatch.
>
> The detector now explicitly returns honest categorical verdicts (`likely_genuine`, `potential_spoof`, or `inconclusive`) with disclaimers that it does not establish proof of caller identity or fraud.

---

## 6. Recommended Next Steps

1. **Do NOT adjust threshold arbitrarily**: Lowering the threshold to 0.75 or 0.80 would merely make this single recording pass while destroying true spoof recall on real deepfake audio.
2. **Train with Data Augmentation & Cross-Domain Audio**:
   - Augment training data with Room Impulse Responses (reverberation), additive noise (MUSAN), and microphone frequency response profiles.
   - Include multi-microphone corpora and diverse recording conditions.
3. **Transition to SOTA Architecture**: Replace toy STFT CNN with pre-trained self-supervised representations (e.g., WavLM / wav2vec 2.0) or AASIST / RawNet2 designed for generalized voice spoof detection.
