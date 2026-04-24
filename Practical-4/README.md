# Practical 4 — Transformer from Scratch using PyTorch

**Name:** Paras Vishwakarma | **Roll No:** 48 | **Class:** C | **BE SEM-II 2025-26**  
**Subject:** (410256) Laboratory Practice VI — NLP  
**College:** Genba Sopanrao Moze College of Engineering

---

## Problem Statement

Implement a complete Transformer model from scratch using the PyTorch library, based on the original paper *"Attention Is All You Need"* (Vaswani et al., 2017). The implementation must include:

1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Positional Encoding
4. Position-wise Feed Forward Network
5. Encoder Layer (with residual connections + Layer Norm)
6. Decoder Layer (with masked attention + cross-attention)
7. Full Transformer (Encoder + Decoder stack)
8. Training on a copy task with loss tracking

---

## Libraries Used

| Library | Purpose |
|---|---|
| `torch` | Core PyTorch framework for building neural networks |
| `torch.nn` | Neural network modules (Linear, Embedding, LayerNorm, etc.) |
| `torch.nn.functional` | Functional API (softmax, relu) |
| `torch.optim` | Adam optimizer |
| `numpy` | Numerical operations |
| `math` | sqrt, log constants |

```bash
pip install torch numpy
```

---

## Theory

### What is a Transformer?
A Transformer is a deep learning architecture based entirely on **attention mechanisms**, without any recurrence (RNN) or convolution (CNN). It processes all tokens in parallel, making it very efficient for long sequences.

### Core Components

#### 1. Scaled Dot-Product Attention
The fundamental building block. Given queries (Q), keys (K), and values (V):

```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

- **Q (Query):** What we're looking for
- **K (Key):** What each token can offer
- **V (Value):** The actual content retrieved
- Scaling by `sqrt(d_k)` prevents vanishing gradients in softmax

#### 2. Multi-Head Attention
Instead of one attention, run `h` attention heads in parallel with different learned projections, then concatenate:

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W_o
head_i = Attention(Q*W_qi, K*W_ki, V*W_vi)
```

Allows the model to attend to different types of relationships simultaneously.

#### 3. Positional Encoding
Since Transformers have no recurrence, position information is injected using sine/cosine functions:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 4. Position-wise Feed Forward Network
A two-layer MLP applied independently to each position:
```
FFN(x) = max(0, x*W1 + b1) * W2 + b2
```
Inner dimension is typically 4× the model dimension.

#### 5. Encoder Layer
```
x = LayerNorm(x + MultiHeadSelfAttention(x))
x = LayerNorm(x + FFN(x))
```

#### 6. Decoder Layer
Has three sub-layers:
1. **Masked Self-Attention** — prevents looking at future tokens
2. **Cross-Attention** — Q from decoder, K & V from encoder output
3. **Feed Forward Network**

Each with residual connection and Layer Norm.

#### 7. Full Transformer
- Stack N Encoder layers → process source sequence
- Stack N Decoder layers → generate target sequence
- Final linear + softmax → vocabulary probabilities

### Transformer vs RNN

| Feature | RNN/LSTM | Transformer |
|---|---|---|
| Parallelism | Sequential (slow) | Fully parallel (fast) |
| Long-range dependencies | Difficult (vanishing gradient) | Excellent (direct attention) |
| Training time | Slow | Fast |
| Used in | Older NLP | GPT, BERT, T5, ChatGPT |

---

## Model Architecture Summary

```
Input Tokens (src)
       │
  Embedding + Positional Encoding
       │
  ┌────┴────┐
  │ Encoder │  × N layers
  │  ├─ Multi-Head Self-Attention
  │  ├─ Add & Norm
  │  ├─ Feed Forward
  │  └─ Add & Norm
  └────┬────┘
       │ enc_output
  ┌────┴────┐
  │ Decoder │  × N layers
  │  ├─ Masked Multi-Head Self-Attention
  │  ├─ Add & Norm
  │  ├─ Cross-Attention (enc_output)
  │  ├─ Add & Norm
  │  ├─ Feed Forward
  │  └─ Add & Norm
  └────┬────┘
       │
  Linear + Softmax
       │
  Output Probabilities (vocab_size)
```

---

## Conclusion

In this practical, we successfully implemented a complete Transformer model from scratch using PyTorch without using any pre-built transformer libraries. We built each component individually — scaled dot-product attention, multi-head attention, positional encoding, encoder/decoder layers — and assembled them into a full Transformer. The model was trained on a copy task and showed consistent loss reduction over epochs. This implementation matches the architecture described in the original *"Attention Is All You Need"* paper and forms the foundation of modern large language models like GPT and BERT.

---

*Submitted by: Paras Vishwakarma | Roll No: 48 | Class: C | BE SEM-II | 2025-26*
