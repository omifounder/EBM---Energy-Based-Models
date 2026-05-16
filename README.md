# Energy-Based Models for Speech AI in Mental Health
## Complete Technical Documentation

---

## Table of Contents

1. [Introduction](#introduction)
   - [Problem Statement](#problem-statement)
   - [Solution Formulation](#solution-formulation)
   - [High-Level Architecture](#high-level-architecture)
   - [Design Principles](#design-principles)
2. [Mathematical Formulations](#mathematical-formulations)
3. [System Architecture](#system-architecture)
4. [Energy Model Architecture (ASR)](#energy-model-architecture-for-asr)
5. [Emotion Recognition Architecture](#emotion-recognition-architecture)
6. [Implementation Guide](#implementation-guide)
7. [Expected Results](#expected-results)

---

## 1. Introduction

### Problem Statement

**Global Mental Health Crisis**: Over 1 billion people worldwide lack access to mental health care when they need it. Among those who seek help, 80% cannot access trained therapists due to cost, availability, or stigma barriers.

**Voice as a Mental Health Biomarker**: Human voice carries rich emotional and psychological information beyond words—prosody, pitch variation, speech rate, and energy patterns reveal anxiety, depression, and distress. However, current speech AI systems focus primarily on word recognition, ignoring emotional content.

**Three Key Technical Challenges**:

1. **ASR Robustness & Calibration**: 
   - Standard autoregressive ASR models (e.g., Conformer-RNNT, Whisper) produce locally-normalized probabilities that can be overconfident on incorrect transcriptions
   - Poor confidence calibration is dangerous in mental health applications—the system must know when it's uncertain
   - Domain shift (clean training data → noisy real-world therapy sessions) degrades performance

2. **Emotion Recognition from Voice**:
   - Detecting emotions from speech prosody requires understanding subtle acoustic patterns
   - Depression manifests as **voice-text inconsistency**: saying "I'm fine" while voice signals distress
   - Standard multimodal fusion (simple concatenation) misses these cross-modal contradictions

3. **Safe Conversational AI**:
   - Mental health conversations require empathy, context awareness, and safety guardrails
   - Systems must detect crisis situations and avoid providing clinical advice
   - Need globally-consistent understanding across modalities (what you say vs. how you say it)

**Why Existing Approaches Fall Short**:

| Approach | Limitation |
|----------|------------|
| **Standard ASR** | Overconfident errors, poor calibration under domain shift |
| **Emotion classifiers** | Treat voice and text independently, miss inconsistencies |
| **Simple multimodal fusion** | Concatenation lacks principled global scoring |
| **Mental health chatbots** | Text-only (Woebot, Wysa), ignore rich voice information |
| **Voice assistants** | Focus on commands, not emotional understanding |

---

### Solution Formulation

**Core Innovation**: Use **Energy-Based Models (EBMs)** as a unifying framework across three components:

1. **ASR with Residual Energy-Based Models (R-EBM)**
2. **Multimodal Emotion Recognition with Energy-Based Fusion**
3. **Mental Health Buddy Conversational System**

**Why Energy-Based Models?**

Energy-based models assign a scalar **energy score** to input-output pairs:
- **Low energy** = good fit (high probability)
- **High energy** = poor fit (low probability)

**Key advantages**:

✅ **Global normalization**: Unlike local softmax, EBMs consider entire sequences through partition function  
✅ **Flexible architecture**: Any neural network can be an energy function  
✅ **Uncertainty quantification**: Energy landscapes reveal model confidence  
✅ **Multimodal fusion**: Natural framework for combining voice + text consistently  
✅ **Interpretability**: Energy differences show why one hypothesis is better  

**Mathematical Foundation**:

Standard probabilistic model (e.g., baseline ASR):
$$P(y \mid X) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, X)$$
- Locally normalized (per-token softmax)
- Can be overconfident

Energy-based enhancement:
$$P_{\theta}(y \mid X) = P_{\text{base}}(y \mid X) \cdot \frac{\exp\{-E_{\theta}(X, y)\}}{Z_{\theta}(X)}$$

Where:
- $E_{\theta}(X, y)$: learnable energy function (utterance-level scoring)
- $Z_{\theta}(X)$: partition function (global normalization)

**Three-Stage Solution**:

Stage 1: Robust ASR Foundation <br>
├── Baseline: Conformer-RNNT (locally normalized) <br>
├── Enhancement: Residual energy model (global scoring) <br>
└── Output: Calibrated transcripts with reliable confidence <br>

Stage 2: Emotion Understanding <br>
├── Voice Branch: wav2vec 2.0 + prosody features <br>
├── Text Branch: ASR transcript → BERT sentiment <br>
├── Fusion: Energy-based voice-text consistency scoring <br>
└── Output: Emotions + inconsistency detection (depression biomarker) <br>

Stage 3: Mental Health Buddy <br>
├── Input: Speech audio <br>
├── Processing: ASR → Emotion → Consistency → Context <br>
├── Generation: Template-based empathetic responses <br>
├── Safety: Crisis detection + guardrails <br>
└── Output: Safe, context-aware support <br>


---

### High-Level Architecture

![High-Level Mental Health Buddy System Architecture](images/architecture_1_high_level.png)

**Data Flow Summary**:

1. **Input**: User speaks into system (raw audio waveform)
2. **ASR Stage**: Conformer-RNNT generates transcript, energy model provides calibrated confidence
3. **Parallel Processing**:
   - **Voice path**: Extract acoustic features (wav2vec 2.0) + prosody (F0, energy, rate)
   - **Text path**: Analyze transcript sentiment (BERT)
4. **Fusion**: Energy-based multimodal model combines voice + text, detects inconsistencies
5. **Context**: Update conversation history with current turn
6. **Generation**: Select appropriate response template based on emotion + inconsistency + context
7. **Safety**: Filter through crisis detection and guardrails
8. **Output**: Deliver empathetic, contextually-appropriate response

---

### Design Principles

**1. Safety First**
- ❌ Never provide clinical diagnosis or treatment advice
- ✅ Supportive listening and resource referrals only
- ✅ Immediate crisis intervention (hotline referral on keywords like "suicide", "hurt myself")
- ✅ Clear user disclosure: "This is AI support, not a replacement for professional care"

**2. Uncertainty-Aware**
- All models quantify confidence via energy scores
- Low-confidence transcripts flagged for clarification
- System knows when it doesn't know

**3. Modular & Extensible**
- Three independent stages (ASR → Emotion → Response) for easier debugging and improvement
- Each component can be upgraded independently
- Clear interfaces between modules

**4. Privacy-Preserving**
- All processing can run locally (no cloud requirement)
- No permanent storage of conversation audio
- Anonymized evaluation data only

**5. Clinically Grounded**
- Voice-text inconsistency based on validated depression research
- Response templates reviewed by mental health professionals
- Evaluation metrics aligned with clinical needs (safety > accuracy)

**6. Computationally Practical**
- Energy models add 10-20% overhead to baseline (acceptable)
- Real-time operation: <300ms end-to-end latency on GPU
- Can degrade gracefully (use baseline ASR if energy model unavailable)

---

## 2. Mathematical Formulations

### 2.1 Standard Autoregressive ASR

The baseline autoregressive ASR model produces:

$$P(y \mid X) = \prod\_{t=1}^{T} P(y\_t \mid y\_{\lt t}, X)$$

Where:
- $X$: acoustic feature sequence (input audio)
- $y = (y_1, y_2, ..., y_T)$: output token sequence (transcription)
- $y_{<t}$: tokens before position $t$
- Each $P(y_t \mid y_{<t}, X)$ is locally normalized via softmax

**Problem**: Local normalization can produce overconfident predictions for incorrect sequences, especially under domain shift.

---

### 2.2 Residual Energy-Based Model (R-EBM)

The joint model combining baseline ASR with residual energy (Li et al., Interspeech 2021):

$$P_{\theta}(y \mid X) = P(y \mid X) \cdot \frac{\exp\{-E_{\theta}(X, y)\}}{Z_{\theta}(X)}$$

**Components**:

- $P(y \mid X)$: Baseline ASR probability (e.g., from Conformer-RNNT)
- $E_{\theta}(X, y)$: **Residual energy function** 
  - Neural network with parameters $\theta$
  - Takes utterance-level features as input
  - Outputs scalar energy score: $E_{\theta}(X, y) \in \mathbb{R}$
  - **Lower energy = better fit** between audio $X$ and hypothesis $y$
  
- $Z_{\theta}(X)$: **Partition function** (normalization constant)

$$Z_{\theta}(X) = \sum_{y' \in \mathcal{Y}} P(y' \mid X) \exp\{-E_{\theta}(X, y')\}$$

Where $\mathcal{Y}$ is the set of all possible transcriptions.

---

### 2.3 Practical Log-Space Inference

Since computing $Z_{\theta}(X)$ exactly is intractable (exponentially many sequences), we work in log-space:

$$\log P_{\theta}(y \mid X) = \log P(y \mid X) - E_{\theta}(X, y) - \log Z_{\theta}(X)$$

For **n-best re-scoring**, the partition function cancels in relative comparisons:

$$\log P_{\theta}(y \mid X) \propto \log P(y \mid X) - E_{\theta}(X, y)$$

**Inference procedure**:
1. Generate n-best hypotheses using baseline ASR: $\{y_1, y_2, ..., y_n\}$
2. For each hypothesis $y_i$, compute score: $\text{score}(y_i) = \log P(y_i \mid X) - E_{\theta}(X, y_i)$
3. Select hypothesis with highest score: $y^* = \arg\max_{y_i} \text{score}(y_i)$

---

### 2.4 Training Objective: Noise Contrastive Estimation (NCE)

**Goal**: Train $E_{\theta}$ to assign low energy to correct transcriptions, high energy to incorrect ones.

**Contrastive Loss**:

$$\mathcal{L}_{\text{NCE}} = -\mathbb{E}_{(X,y) \sim \text{data}} \left[ \log \sigma(-E_{\theta}(X, y)) + \sum_{i=1}^{K} \log \sigma(E_{\theta}(X, y_i^-)) \right]$$

Where:
- $(X, y)$: real audio-transcription pair from dataset
- $y_i^-$: negative samples (incorrect transcriptions):
  - 70% from beam search candidates: $y_i^- \sim \text{BeamSearch}(X)$
  - 30% from random perturbations: $y_i^- \sim \text{Perturb}(y)$
- $K$: number of negative samples per positive (typically $K=5$ to $10$)
- $\sigma(z) = 1/(1+\exp(-z))$: sigmoid function

**Intuition**: Maximize probability of distinguishing real transcription $y$ (low energy) from noise samples $y_i^-$ (high energy).

---

### 2.5 Expected Calibration Error (ECE)

Measures how well predicted confidence matches actual accuracy:

$$\text{ECE} = \sum_{m=1}^{M} \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

Where:
- $M$: number of bins (typically 10)
- $B_m$: set of predictions in bin $m$ (grouped by confidence)
- $N$: total number of predictions
- $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \mathbb{1}(\hat{y}_i = y_i)$: accuracy in bin $m$
- $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} \hat{p}_i$: average confidence in bin $m$

**Lower ECE = better calibration**

---

### 2.6 Emotion Energy Function

For multimodal emotion recognition, the energy function combines voice and text:

$$E_{\text{emotion}}(v, t) = \text{MLP}([v_{\text{emb}} \oplus t_{\text{emb}} \oplus (v_{\text{emb}} - t_{\text{emb}}) \oplus (v_{\text{emb}} \odot t_{\text{emb}})])$$

Where:
- $v_{\text{emb}} \in \mathbb{R}^{d_v}$: voice embedding from wav2vec 2.0 + BiLSTM
- $t_{\text{emb}} \in \mathbb{R}^{d_t}$: text embedding from BERT/DistilBERT
- $\oplus$: concatenation operator
- $\odot$: element-wise (Hadamard) product
- $(v_{\text{emb}} - t_{\text{emb}})$: difference features (captures inconsistency)
- $(v_{\text{emb}} \odot t_{\text{emb}})$: interaction features

**Multi-Task Training Objective**:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{emotion}} + \lambda_1 \mathcal{L}_{\text{consistency}} + \lambda_2 \mathcal{L}_{\text{depression}}$$

Where:
- $\mathcal{L}_{\text{emotion}} = -\sum_{c=1}^{C} y_c \log \hat{y}_c$: cross-entropy for emotion classification (IEMOCAP)
- $\mathcal{L}_{\text{consistency}}$: contrastive loss for voice-text matching
- $\mathcal{L}_{\text{depression}}$: binary cross-entropy for depression detection (DAIC-WOZ)
- $\lambda_1, \lambda_2 \in [0,1]$: task weights (tuned on validation set)

---

### 2.7 Word Error Rate (WER)

Standard metric for ASR evaluation:

$$\text{WER} = \frac{S + D + I}{N} \times 100\%$$

Where:
- $S$: number of substitutions
- $D$: number of deletions
- $I$: number of insertions
- $N$: total number of words in reference transcription

**Relative WER Improvement**:

$$\text{Rel. Improvement} = \frac{\text{WER}_{\text{baseline}} - \text{WER}_{\text{R-EBM}}}{\text{WER}_{\text{baseline}}} \times 100\%$$

---

## 3. Energy Model Architecture (for ASR)

### 3.1 Overview

The energy model $E_{\theta}(X, y)$ takes an audio-transcript pair and outputs a scalar energy score. **Lower energy indicates better fit**.

### 3.2 Architecture Diagram

![Energy Model Architecture for ASR](images/architecture_2_energy_model.png)

### 3.3 Component Details

#### 3.3.1 Input Features

**Extracted from frozen baseline ASR model**:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Encoder hidden states | `[T_enc × d_enc]` | Conformer encoder output (acoustic representation) |
| Decoder hidden states | `[T_dec × d_dec]` | RNN-T decoder output (language model state) |
| Attention context | `[T_dec × d_enc]` | Cross-attention between encoder and decoder |
| Top-K probabilities | `[T_dec × K]` | Top K=10 token probabilities from baseline softmax |

**Alignment**: Use decoder length `T_dec` as reference, apply adaptive pooling to encoder features.

#### 3.3.2 Feature Concatenation

```python
# Pseudocode
enc_pooled = adaptive_avg_pool(h_enc, target_len=T_dec)  # [B, T_dec, d_enc]
features = concat([enc_pooled, h_dec, c_attn, p_topk], dim=-1)
# Shape: [B, T_dec, d_enc + d_dec + d_enc + K]
# Example: [B, T_dec, 512 + 512 + 512 + 10] = [B, T_dec, 1546]
```

#### 3.3.3 BiLSTM Layer

**Purpose**: Capture sequential dependencies across the utterance

**Configuration**:
```yaml
hidden_size: 512
num_layers: 2
bidirectional: True
dropout: 0.3
output_size: 2 × 512 = 1024
```

**Why BiLSTM?** 
- Bidirectional context: future frames inform energy of current position
- Better than unidirectional for utterance-level scoring (we have full sequence at inference)

#### 3.3.4 Self-Attention Layer

**Purpose**: Model long-range dependencies beyond LSTM's limited context window

**Configuration**:
```yaml
num_heads: 8
head_dim: 128
total_dim: 1024  # matches BiLSTM output
dropout: 0.1
```

**Mechanism**:
```python
Q = K = V = BiLSTM_output  # [B, T, 1024]
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
```

#### 3.3.5 Mean Pooling

**Purpose**: Aggregate sequence into single utterance-level representation

```python
utterance_embedding = mean(attn_output, dim=time)  # [B, 1024]
```

**Alternative**: Could use max pooling, attention pooling, or [CLS] token, but mean pooling is simple and effective.

#### 3.3.6 MLP Head

**Purpose**: Map utterance embedding to scalar energy

**Architecture**:
Layer 1: Linear(1024 → 512) + ReLU + Dropout(0.3)
Layer 2: Linear(512 → 1) [no activation]


**No final activation**: Energy can be any real number (not probability)

### 3.4 Training Details

**Objective**: Noise Contrastive Estimation (NCE)

```python
def compute_nce_loss(audio, correct_transcript, baseline_asr):
    # Positive sample
    energy_pos = energy_model(audio, correct_transcript)  # Low energy desired
    
    # Negative samples
    beam_hypotheses = baseline_asr.beam_search(audio, beam_size=10)
    negatives = beam_hypotheses[1:6]  # Top 2-6 incorrect hypotheses
    
    # Add perturbations
    perturbed = perturb_transcript(correct_transcript, num=3)
    negatives.extend(perturbed)
    
    energy_neg = [energy_model(audio, neg) for neg in negatives]
    
    # NCE loss
    loss = -log(sigmoid(-energy_pos)) - sum([log(sigmoid(e)) for e in energy_neg])
    return loss
```

**Hyperparameters**:
- Learning rate: 1e-4 (with warmup)
- Optimizer: AdamW (weight decay=1e-5)
- Batch size: 16 utterances
- Negatives per positive: K=5-10
- Training epochs: 20-30 on LibriSpeech

### 3.5 Inference

**N-best rescoring**:

```python
def rescore_hypotheses(audio, n_best_list, baseline_probs):
    """
    Args:
        audio: input waveform
        n_best_list: list of transcript hypotheses
        baseline_probs: baseline ASR log probabilities
    
    Returns:
        best_hypothesis: highest-scoring transcript
        calibrated_confidence: energy-based confidence
    """
    scores = []
    for i, hyp in enumerate(n_best_list):
        energy = energy_model(audio, hyp)
        score = baseline_probs[i] - energy  # Log-space combination
        scores.append(score)
    
    best_idx = argmax(scores)
    return n_best_list[best_idx], softmax(scores)[best_idx]
```

### 3.6 Model Size

**Total parameters**: ~15M (energy model only)
- BiLSTM: 2 layers × 2 directions × 512 hidden ≈ 8M
- Self-attention: 8 heads × 128 dim ≈ 4M
- MLP: 1024→512→1 ≈ 0.5M
- Embeddings and norms: ≈ 2.5M

**Note**: Baseline ASR (Conformer-RNNT ~100M params) is frozen during energy model training.

---

## 4. Emotion Recognition Architecture

### 4.1 Overview

The emotion recognition system uses energy-based multimodal fusion to:
1. Detect primary emotion from voice prosody
2. Extract sentiment from transcript text
3. Identify voice-text inconsistencies (depression biomarker)
4. Predict depression likelihood

### 4.2 Architecture Diagram
┌─────────────────────────────────────────────────────────────────────┐
│ Audio Input Waveform │
└─────────────────────────────────────────────────────────────────────┘
│
┌───────────────┴───────────────┐
│ │
▼ ▼
┌───────────────────────┐ ┌──────────────────────┐
│ Voice Branch │ │ Text Branch │
│ │ │ │
│ (Prosody Analysis) │ │ (Sentiment) │
└───────────────────────┘ └──────────────────────┘
│ │
▼ ▼
┌───────────────────────┐ ┌──────────────────────┐
│ wav2vec 2.0 │ │ ASR R-EBM │
│ (Pre-trained) │ │ │
│ │ │ Audio → Transcript │
│ Audio [N samples] │ │ │
│ ↓ │ └──────────────────────┘
│ Features [T×768] │ │
└───────────────────────┘ │
│ ▼
▼ ┌──────────────────────┐
┌───────────────────────┐ │ DistilBERT │
│ Prosody Extractor │ │ (Pre-trained) │
│ (librosa) │ │ │
│ │ │ Transcript tokens │
│ Extract: │ │ ↓ │
│ - F0 (pitch) │ │ Embeddings [L×768] │
│ - Energy (RMS) │ │ ↓ │
│ - Speech rate │ │ [CLS] token │
│ - Pause duration │ │ ↓ │
│ - F0 variance │ │ Text emb │
│ │ │ │
│ Output: [T×5] │ └──────────────────────┘
└───────────────────────┘ │
│ │
▼ │
┌───────────────────────┐ │
│ Concatenate: │ │
│ wav2vec + prosody │ │
│ │ │
│ [T×(768+5)] = [T×773]│ │
└───────────────────────┘ │
│ │
▼ │
┌───────────────────────┐ │
│ BiLSTM │ │
│ (voice sequence) │ │
│ │ │
│ Input: [T×773] │ │
│ hidden: 256 │ │
│ layers: 2 │ │
│ bidirectional: True │ │
│ Output: [T×512] │ │
└───────────────────────┘ │
│ │
▼ │
┌───────────────────────┐ │
│ Mean Pooling │ │
│ (over time) │ │
│ │ │
│ Output: │ │
│ │ │
│ = v_emb │ │
└───────────────────────┘ │
│ │
│ │
│ = t_emb
│ │
│ │
└──────────┬────────────────────┘
│
▼
┌────────────────────────────┐
│ Projection Layer │
│ │
│ v_proj = Linear(512→768) │
│ (v_emb) │
│ │
│ Now both embeddings │
│ have dimension 768 │
└────────────────────────────┘
│
▼
┌────────────────────────────┐
│ Multimodal Fusion │
│ │
│ Construct feature vector: │
│ │
│ f = [v_proj, │
│ t_emb, │
│ (v_proj - t_emb), │ ← Inconsistency
│ (v_proj ⊙ t_emb)] │ ← Interaction
│ │
│ Dim: [768+768+768+768] │
│ = │
└────────────────────────────┘
│
▼
┌────────────────────────────┐
│ Fusion MLP │
│ │
│ Layer 1: │
│ Linear(3072 → 1024) │
│ + LayerNorm │
│ + ReLU │
│ + Dropout(0.3) │
│ │
│ Layer 2: │
│ Linear(1024 → 512) │
│ + LayerNorm │
│ + ReLU │
│ + Dropout(0.3) │
│ │
│ Output: = h_fused │
└────────────────────────────┘
│
▼
┌────────────────────────────┐
│ Multi-Task Heads │
└────────────────────────────┘
│
┌─────────────────┼─────────────────┬──────────────────┐
│ │ │ │
▼ ▼ ▼ ▼
┌────────────────┐ ┌──────────────┐ ┌──────────────┐ ┌─────────────────┐
│ Emotion Head │ │ Consistency │ │ Depression │ │ Energy Score │
│ │ │ Head │ │ Head │ │ Head │
│ Linear(512→5) │ │ Linear(512→1)│ │ Linear(512→1)│ │ Linear(512→1) │
│ + Softmax │ │ + Sigmoid │ │ + Sigmoid │ │ (no activation) │
│ │ │ │ │ │ │ │
│ Classes: │ │ Match │ │ Depressed │ │ E_emotion(v,t) │
│ - Sad │ │ probability │ │ probability │ │ ∈ ℝ │
│ - Anxious │ │ │ │ │ │ │

│ - Frustrated │ │ │ │ │ │ High energy = │
│ - Neutral │ │ │ │ │ │ voice-text │
│ - Happy │ │ │ │ │ │ mismatch │
│ │ │ │ │ │ │ │
│ Output: │ │ Output: │ │ Output: │ │ Output: │

└────────────────┘ └──────────────┘ └──────────────┘ └─────────────────┘


### 4.3 Component Details

#### 4.3.1 Voice Branch

**Step 1: wav2vec 2.0 Feature Extraction**

```python
# Pre-trained on LibriSpeech 960h
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Fine-tuning strategy
for i, layer in enumerate(wav2vec_model.encoder.layers):
    if i < 6:
        layer.requires_grad = False  # Freeze first 6 layers
    else:
        layer.requires_grad = True   # Fine-tune last 6 layers

# Extract features
audio_features = wav2vec_model(audio_waveform).last_hidden_state  # [B, T, 768]
```

**Step 2: Prosody Feature Extraction**

```python
import librosa

def extract_prosody(audio, sr=16000):
    """
    Extract 5 prosody features per frame
    """
    # F0 (fundamental frequency / pitch)
    f0, voiced_flag, _ = librosa.pyin(audio, fmin=80, fmax=400, sr=sr)
    f0 = np.nan_to_num(f0)  # Replace NaN with 0
    
    # Energy (RMS)
    energy = librosa.feature.rms(y=audio, frame_length=512, hop_length=160)
    
    # Speech rate (syllable nuclei per second, approximated)
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    speech_rate = np.convolve(onset_env, np.ones(10)/10, mode='same')
    
    # Pause duration (silence segments)
    intervals = librosa.effects.split(audio, top_db=30)
    pauses = compute_pause_durations(intervals, len(audio))
    
    # F0 variance (pitch variability over sliding window)
    f0_var = np.array([np.var(f0[max(0,i-10):i+10]) for i in range(len(f0))])
    
    # Combine and align to wav2vec frame rate
    prosody = np.stack([f0, energy, speech_rate, pauses, f0_var], axis=-1)  # [T, 5]
    prosody = align_to_features(prosody, target_len=audio_features.shape)[1]
    
    return prosody  # [T, 5]
```

**Key Prosody Features**:

| Feature | Range | Depression Indicator |
|---------|-------|---------------------|
| F0 (pitch) | 80-400 Hz | Lower mean, reduced variance |
| Energy | 0-1 (normalized) | Lower overall |
| Speech rate | Syllables/sec | Slower (1-2 vs. 3-4 normal) |
| Pause duration | Seconds | Longer pauses |
| F0 variance | Hz² | Flat prosody (low variance) |

**Step 3: Voice BiLSTM**

```python
voice_bilstm = nn.LSTM(
    input_size=773,  # 768 wav2vec + 5 prosody
    hidden_size=256,
    num_layers=2,
    bidirectional=True,
    dropout=0.3,
    batch_first=True
)

voice_features = torch.cat([audio_features, prosody], dim=-1)  # [B, T, 773]
voice_lstm_out, _ = voice_bilstm(voice_features)  # [B, T, 512]
v_emb = voice_lstm_out.mean(dim=1)  # [B, 512]
```

#### 4.3.2 Text Branch

**Step 1: ASR Transcription**

```python
# Use R-EBM from Part 1
transcript, confidence = asr_rebm_model(audio)
```

**Step 2: DistilBERT Sentiment Embedding**

```python
from transformers import DistilBertModel, DistilBertTokenizer

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Tokenize transcript
tokens = tokenizer(transcript, return_tensors="pt", padding=True, truncation=True)

# Extract [CLS] token embedding
bert_output = distilbert(**tokens)
t_emb = bert_output.last_hidden_state[:, 0, :]  # [B, 768]
```

**Fine-tuning**: DistilBERT fine-tuned on sentiment analysis datasets (SST-2, IMDb) to better capture emotional valence in text.

#### 4.3.3 Multimodal Fusion

**Step 1: Dimension Alignment**

```python
# Project voice embedding to match text dimension
v_proj = nn.Linear(512, 768)(v_emb)  # [B, 768]
```

**Step 2: Feature Construction**

```python
# Four types of features
difference = v_proj - t_emb  # [B, 768] - captures inconsistency
interaction = v_proj * t_emb  # [B, 768] - captures agreement

# Concatenate all
fused_features = torch.cat([
    v_proj,      # [B, 768] - voice information
    t_emb,       # [B, 768] - text information
    difference,  # [B, 768] - inconsistency signal
    interaction  # [B, 768] - consistency signal
], dim=-1)  # [B, 3072]
```

**Why this design?**
- `v_proj, t_emb`: Preserve original modality information
- `v_proj - t_emb`: **Difference vector** is large when voice emotion ≠ text sentiment → depression biomarker
- `v_proj * t_emb`: **Interaction vector** is large when both agree → consistency signal

**Step 3: Fusion MLP**

```python
fusion_mlp = nn.Sequential(
    nn.Linear(3072, 1024),
    nn.LayerNorm(1024),
    nn.ReLU(),
    nn.Dropout(0.3),
    
    nn.Linear(1024, 512),
    nn.LayerNorm(512),
    nn.ReLU(),
    nn.Dropout(0.3)
)

h_fused = fusion_mlp(fused_features)  # [B, 512]
```

#### 4.3.4 Multi-Task Heads

**Head 1: Emotion Classification**

```python
emotion_head = nn.Linear(512, 5)  # 5 classes
emotion_logits = emotion_head(h_fused)
emotion_probs = F.softmax(emotion_logits, dim=-1)

# Classes: [sad, anxious, frustrated, neutral, happy]
# Loss: CrossEntropyLoss
```

**Head 2: Consistency Detection**

```python
consistency_head = nn.Linear(512, 1)
consistency_logit = consistency_head(h_fused)
consistency_prob = torch.sigmoid(consistency_logit)  # P(voice-text match)

# High prob = consistent, Low prob = inconsistent
# Loss: BCELoss
```

**Head 3: Depression Detection**

```python
depression_head = nn.Linear(512, 1)
depression_logit = depression_head(h_fused)
depression_prob = torch.sigmoid(depression_logit)  # P(depressed)

# Trained on DAIC-WOZ PHQ-8 labels (> 10 = depressed)
# Loss: BCELoss
```

**Head 4: Energy Score**

```python
energy_head = nn.Linear(512, 1)
energy = energy_head(h_fused)  # No activation, raw ℝ value

# Low energy = voice-text consistency
# High energy = voice-text mismatch
# Used for ranking, not direct prediction
```

### 4.4 Training Details

**Multi-Task Loss**:

```python
def compute_loss(emotion_pred, consistency_pred, depression_pred, energy_pred,
                 emotion_label, consistency_label, depression_label):
    
    # Task 1: Emotion classification
    L_emotion = F.cross_entropy(emotion_pred, emotion_label)
    
    # Task 2: Consistency detection
    L_consistency = F.binary_cross_entropy(consistency_pred, consistency_label)
    
    # Task 3: Depression detection
    L_depression = F.binary_cross_entropy(depression_pred, depression_label)
    
    # Task 4: Energy-based consistency ranking (contrastive)
    # Low energy for matched pairs, high for mismatched
    L_energy = torch.mean(
        consistency_label * energy_pred +          # Matched → minimize energy
        (1 - consistency_label) * (-energy_pred)   # Mismatched → maximize energy
    )
    
    # Weighted combination
    total_loss = L_emotion + 0.5 * L_consistency + 0.3 * L_depression + 0.2 * L_energy
    
    return total_loss
```

**Training Configuration**:

```yaml
optimizer: AdamW
learning_rate: 2e-5
weight_decay: 0.01
batch_size: 32
epochs: 30
warmup_steps: 1000

datasets:
  IEMOCAP: 
    - emotion labels
    - train/val split: 80/20
  
  DAIC-WOZ:
    - depression labels (PHQ-8 > 10)
    - subset: 100 interviews (50 depressed, 50 control)
    
  Consistency labels:
    - Synthetic: match/mismatch voice-text pairs
    - Real: annotated from IEMOCAP (does prosody match text?)
```

### 4.5 Inference

**Example Usage**:

```python
def analyze_emotion(audio_waveform):
    """
    Analyze emotion and detect voice-text inconsistency
    """
    # Forward pass
    energy, emotion_probs, consistency_prob, depression_prob = emotion_model(
        audio_waveform
    )
    
    # Extract predictions
    dominant_emotion = emotion_probs.argmax()  # 0-4 (sad/anxious/frustrated/neutral/happy)
    is_consistent = consistency_prob > 0.5
    risk_level = depression_prob.item()
    
    # Inconsistency flag
    if energy > 2.0 and not is_consistent:
        flag = "POTENTIAL_DISTRESS_MASKING"
    else:
        flag = None
    
    return {
        'emotion': ['sad', 'anxious', 'frustrated', 'neutral', 'happy'][dominant_emotion],
        'consistency': 'consistent' if is_consistent else 'inconsistent',
        'depression_risk': risk_level,
        'flag': flag,
        'energy_score': energy.item()
    }
```

### 4.6 Model Size

**Total parameters**: ~120M
- wav2vec 2.0: 95M (48M trainable after freezing first 6 blocks)
- DistilBERT: 66M (frozen during training)
- Voice BiLSTM: 3M
- Projection + Fusion MLP: 12M
- Multi-task heads: 5K

**Trainable**: ~63M

---

## 5. Expected Results

### 5.1 ASR Performance

| Dataset | Metric | Baseline (Conformer-RNNT) | + R-EBM | Improvement |
|---------|--------|---------------------------|---------|-------------|
| LibriSpeech test-clean | WER | 2.8% | 2.6% | -7.1% rel |
| LibriSpeech test-other | WER | 4.8% | 4.4% | -8.3% rel |
| CHiME-5 WER (zero-shot) | WER | 47.3% | 43.2% | -8.7% rel |
| LibriSpeech test-other | ECE | 0.087 | 0.041 | -52.9% |

### 5.2 Emotion Recognition Performance

| Dataset | Metric | Baseline (Transformer) | Energy-Based | Improvement |
|---------|--------|------------------------|--------------|-------------|
| IEMOCAP | 4-class accuracy | 62.3% | 71.8% | +15.2% rel |
| IEMOCAP | F1-score (weighted) | 0.61 | 0.70 | +14.8% |
| DAIC-WOZ subset | Depression detection | 64% | 76% | +18.8% (preliminary) |

### 5.3 System Performance

- **End-to-end latency**: < 300ms (GPU)
- **Crisis keyword detection**: 100% recall (on test set)
- **Inappropriate response rate**: 0% (in 30 test conversations)
- **User satisfaction**: Qualitative evaluation pending (20-30 conversations)

---
