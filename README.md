# LogicBot

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An experimental fine-tuning project exploring how small language models behave on custom instruction datasets — and what happens when data is scarce.

---

## What This Project Is
<img width="1582" height="730" alt="image" src="https://github.com/user-attachments/assets/a8f58cc3-0d05-4e2f-8961-2df5da0e36ed" />


<img width="1582" height="730" alt="LogicBot v2 running in terminal" src="https://github.com/user-attachments/assets/baf3695f-202d-4936-8da5-9c61423d7df6" />

LogicBot is a learning prototype. It fine-tunes **DialoGPT-small** on a custom dataset of logical and philosophical question-answer pairs to explore the mechanics and limits of instruction tuning in low-resource settings.

The project is **not** a production chatbot. It is an honest investigation into what language models can and cannot learn from limited data, and why.

---

## Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| **PyTorch** | Model training, loss computation, backpropagation |
| **HuggingFace Transformers** | Model loading, tokenizer, `Trainer` API |
| **DialoGPT-small** (Microsoft) | Base pretrained model (~117M parameters) |
| **Datasets** (HuggingFace) | JSONL dataset loading and preprocessing |
| **AdamW Optimizer** | Weight-decay regularized gradient descent |
| **Python 3.8+** | Core implementation language |

---

## Model Architecture

**Base model:** `microsoft/DialoGPT-small`

DialoGPT is a GPT-2 variant trained on Reddit conversation data. Its architecture is a **Transformer Decoder** — a stack of self-attention blocks that model token sequences autoregressively.

The model predicts each token conditioned on all preceding tokens:

```
P(x_t | x_1, x_2, ..., x_{t-1}; θ)
```

Where:
- `x_t` is the token at position `t`
- `θ` represents all learnable parameters (~117M weights)
- The prediction is computed via softmax over the full vocabulary

At inference, this factorization means the model generates one token at a time, left to right.

---

## Transformer Internals

This section explains what happens *inside* each decoder layer — the mathematical components that make attention-based models work.

### Scaled Dot-Product Self-Attention

Each token attends to every other token by computing three linear projections from the input embeddings `X`:

```
Q = X·W_Q       (Query  — what this token is looking for)
K = X·W_K       (Key    — what each token offers to match against)
V = X·W_V       (Value  — what each token contributes to output)
```

The attention output is:

```
                    ⎛  Q·Kᵀ  ⎞
Attention(Q,K,V) = softmax⎜ ——————— ⎟ · V
                    ⎝  √d_k  ⎠
```

The `√d_k` scaling is not cosmetic — without it, the dot products grow proportionally to `d_k`, which pushes the softmax into near-zero gradient regions and stalls learning.

The result is a weighted sum of all value vectors, where the weights express how relevant each token is to the current one. A token at position 50 can directly attend to position 1 — no information bottleneck, no vanishing signal across distance.

### Multi-Head Attention

Rather than computing one global attention map, the model runs `h` heads in parallel, each operating in a lower-dimensional subspace:

```
head_i = Attention(X·W_Qi, X·W_Ki, X·W_Vi)

MHA(X) = Concat(head_1, ..., head_h) · W_O
```

Each head independently learns to attend to different relational patterns — syntax, topic, coreference, proximity — which are concatenated and projected back to model dimension via `W_O`.

DialoGPT-small uses `h = 12` heads, `d_model = 768`, so each head works in a 64-dimensional subspace.

### Position-Wise Feedforward Network

After attention, each token's representation passes independently through a two-layer MLP:

```
FFN(x) = max(0, x·W_1 + b_1) · W_2 + b_2
```

The hidden dimension is `4 × d_model = 3072` for DialoGPT-small — expanding then compressing the representation. ReLU introduces nonlinearity.

**The distinction most people miss:** attention *mixes information across tokens* (who attends to whom). FFN *transforms each token's representation independently* (what does that information mean). Attention without FFN is a purely linear operation. Both are required.

### Residual Connections and Layer Normalization

Every sublayer — attention and FFN — is wrapped with a residual connection:

```
x' = x + Sublayer(x)
```

Followed by Layer Normalization:

```
         x - μ
LN(x) = ———————— · γ + β
           σ
```

Where `μ` and `σ` are computed across the feature dimension, and `γ`, `β` are learnable per-feature scale and shift parameters.

Residual connections create a **gradient highway**: the loss gradient flows directly back to early layers without passing through many nonlinearities. Without them, 12-layer transformers suffer from vanishing gradients and fail to train. LayerNorm keeps activations in a stable numerical range across layers, preventing representation drift with depth.

---

## Training Method

Fine-tuning adapts the pretrained weights toward a new task using supervised learning on labeled examples.

**Input format:**
```
Question: <prompt>
Answer: <response>
```

**Objective — Cross-Entropy Loss:**

```
L(θ) = − Σ_t  log P(y_t | y_<t, x; θ)
```

Where:
- `x` = input prompt tokens
- `y` = target response tokens
- `T` = sequence length
- The model is penalized proportionally to how surprised it is by each correct token

The gradient of this loss is backpropagated through all transformer layers to update `θ`.

**Why this works (in theory):** by minimizing cross-entropy over many examples, the model adjusts its internal representations to favor the patterns in the training data.

**Why this has limits here:** with ~140 examples, those patterns are too sparse to generalize. The model fits the training distribution, not the underlying task.

---

## Dataset

| Property | Value |
|----------|-------|
| Format | JSONL |
| Size | ~140 instruction-response pairs |
| Domain | Logic, philosophy, definitions |
| Split | Training only (no held-out validation set) |

**Example entry:**
```json
{
  "prompt": "What is entropy?",
  "response": "Entropy measures disorder in thermodynamic systems."
}
```

### Why Dataset Size Is the Core Problem

| Dataset Size | Expected Behavior |
|--------------|------------------|
| < 500 | Memorization of surface patterns |
| 1k – 5k | Early pattern learning, limited generalization |
| 10k+ | Style-level generalization begins |
| 100k+ | Semantic reasoning starts to emerge |

The current dataset satisfies:

```
|D|  <<  |D_critical|,    where  |D_critical| ≈ 10⁴
```

The model cannot learn abstract relationships — it learns which words statistically follow which other words in the 140 training examples. That is the honest description.

---

## Training Configuration

```python
EPOCHS     = 10
BATCH_SIZE = 2
LR         = 2e-5
MAX_LEN    = 256
OPTIMIZER  = AdamW(weight_decay=0.01)
```

**AdamW** extends Adam with decoupled weight decay. The parameter update rule is:

```
             m̂_t
θ_{t+1} = θ_t − α · ⎛ ————————— + λ·θ_t ⎞
             ⎝ √v̂_t + ε          ⎠
```

Where `m̂_t` and `v̂_t` are bias-corrected first and second moment estimates of the gradient, and `λ` is the weight decay coefficient. Weight decay penalizes large parameter values — L2 regularization decoupled from the adaptive learning rate.

---

## Inference

Text generation uses **nucleus (top-p) sampling**. The model samples from the smallest set of tokens whose cumulative probability exceeds `p`:

```
P_sample(x_t)  ∝  P(x_t) · 𝟙[x_t ∈ TopP(p)]
```

```python
temperature = 0.6   # Sharpens the distribution (lower = less random)
top_p       = 0.9   # Truncates the low-probability tail
```

**Effect of temperature** on the softmax distribution:

```
              exp(log P(x) / T)
P_scaled(x) = ——————————————————————
              Σ_{x'} exp(log P(x') / T)
```

```
T < 1  →  distribution sharpened, favors high-probability tokens
T > 1  →  distribution flattened, more uniform sampling
T → 0  →  equivalent to greedy decoding (argmax)
```

---

## Logic Mode System

A rule-based classifier routes questions to reasoning categories before generation:

```python
LOGIC_MODES = {
    "formal":        ["prove", "theorem", "axiom", "definition"],
    "philosophical": ["meaning", "consciousness", "ethics", "existence"],
    "causal":        ["why", "because", "cause", "reason"],
}
```

This is a **heuristic keyword matcher**. It adjusts the prompt prefix passed to the model, which weakly steers output style. The model has no symbolic reasoning module — the routing is entirely surface-level.

---

## Reasoning and Chain-of-Thought

### What "Reasoning" Actually Means in Autoregressive Models

Language models do **not** perform symbolic reasoning. They do not maintain a logic engine, proof state, or working memory. Every output token is computed identically — attention over the context window followed by softmax over the vocabulary.

What is called "reasoning" is more precisely: **conditional token generation that statistically mimics reasoning traces seen during training.**

The base probability factorization is unchanged regardless of how complex the output appears:

```
P(y | x) = ∏_t  P(y_t | y_<t, x; θ)
```

### Chain-of-Thought as Latent Trajectory Sampling

When a model generates intermediate steps before the final answer, it produces a reasoning trace `r_1, r_2, ..., r_n` as part of the output sequence:

```
x  →  r_1  →  r_2  →  ...  →  y
```

This can be written as marginalizing over latent reasoning paths:

```
P(y | x)  ≈  Σ_r  P(y | r, x) · P(r | x)
```

Each intermediate token `r_i` conditions subsequent generation, decomposing a hard prediction into smaller steps. This works when the training data contains enough structured explanation that intermediate tokens reliably correlate with correct final tokens.

**The honest framing:** chain-of-thought is latent trajectory sampling in token space. It is a statistical phenomenon, not a cognitive one. It emerges from training distribution structure — not from the model "thinking."

This technique, studied formally by Wei et al. (2022), was later scaled aggressively. **DeepSeek-R1** extended this by training models to produce explicit `<think>...</think>` reasoning traces — long internal monologues before giving a final answer. The model is rewarded for traces that lead to correct outputs via reinforcement learning, making the reasoning process itself a learned behavior rather than a prompted one. The underlying math is the same trajectory sampling — but the training signal now optimizes the paths `r_i`, not just the answer `y`.

LogicBot's training data contains no reasoning traces, so chain-of-thought behavior is not expected to emerge here.

### Chain-of-Thought Caching (How DeepSeek Optimizes It)

A practical problem with long reasoning traces: they are expensive. If the same reasoning path `r_1, ..., r_n` is reused across queries (e.g., a fixed problem-solving template), recomputing attention over those tokens every time is wasteful.

**KV-Cache** stores the key and value tensors for already-computed tokens:

```
At step t,  K_t and V_t are computed once and cached.

For step t+1:  only the new token's Q is computed.
               K and V are retrieved from cache.
```

For a sequence of length `T`, this reduces attention computation from `O(T²)` per step to `O(T)` per new token after the prefix is cached.

DeepSeek and similar systems extend this to **prefix caching** — if a reasoning preamble is shared across many requests, its KV state is cached and reused. This makes long chain-of-thought practical at inference scale, not just theoretically interesting.

```
Standard inference:    recompute full attention at every step
Prefix KV-cache:       compute prefix once, cache K/V, reuse across queries

Cost reduction:        from O(n·T²) to O(T²) + O(n·t_new)
```

This architectural optimization is what makes reasoning-heavy models like DeepSeek-R1 deployable in production — without it, long reasoning traces would be prohibitively slow.

### Self-Reflection and Iterative Refinement

Some systems implement iterative generation where the model revises its own output:

```
y_0      = f_θ(x)
y_{t+1}  = f_θ(x, y_t)
```

In fixed-point terms, this searches for:

```
y* = f_θ(x, y*)
```

Used in self-consistency decoding (sampling multiple reasoning paths and majority-voting the answer) and Constitutional AI-style pipelines. Not implemented in LogicBot, but represents a concrete architectural direction for v2.

---

## Why the Model Underperforms

These are the concrete technical reasons — not excuses.

### 1. Data scarcity (primary cause)

140 samples cannot cover the input distribution the model encounters at inference. The model overfits:

```
L_train  ↓   while   L_test  ↑
```

Parameters converge to memorized training fragments, not generalizable representations.

### 2. No instruction pretraining

DialoGPT was pretrained on casual Reddit conversation, not instruction-following. Fine-tuning 140 examples on a mismatched pretraining distribution cannot produce task-aligned behavior.

### 3. No RLHF or preference optimization

The training signal is next-token prediction loss only. No reward model, no human preference data, no KL penalty to stabilize outputs. Outputs are entirely unfiltered.

### 4. No retrieval component

All knowledge must exist in the model weights. There is no external memory:

```
Knowledge ⊆ θ
```

The model cannot look anything up. Factual accuracy is bounded by pretrained representations, partially overwritten by fine-tuning.

### 5. Known failure modes

| Failure Mode | Cause |
|---|---|
| Token repetition (`computable.computable...`) | Mode collapse in low-entropy states |
| Semantic drift | Attention not anchored to prompt after several tokens |
| Hallucination | No factual grounding, confident sampling |
| Incoherent definitions | Vocabulary present, logical structure absent |

These are expected behaviors in undertrained autoregressive models — not implementation bugs.

---

## What This Project Does Demonstrate

Despite the limitations, this project covers real ML engineering end-to-end:

- Tokenization pipeline and padding strategy
- JSONL dataset loading and preprocessing
- HuggingFace `Trainer` API configuration
- Gradient accumulation with small batch sizes
- Nucleus sampling inference with temperature control
- Prompt engineering for instruction-style models
- Quantitative failure analysis

Understanding precisely *why* a model fails is as technically valuable as making one succeed.

---

## Architectural Directions to Enhance This System

These are not vague future goals — they are specific techniques with known mathematical properties that directly address the failure modes above.

### LoRA — Parameter-Efficient Fine-Tuning

Instead of updating all 117M parameters, LoRA injects trainable low-rank decomposition matrices into the attention weight projections:

```
W' = W_0 + ΔW = W_0 + B·A

    where  B ∈ ℝ^{d×r},  A ∈ ℝ^{r×k},  r << min(d, k)
```

Only `B` and `A` are trained — `W_0` is frozen. For rank `r=8` on a 768×768 projection, this is `8×(768+768) = 12,288` trainable parameters versus `589,824` for the full matrix. A ~48× reduction.

This directly addresses overfitting: fewer trainable parameters means the model cannot memorize the 140 examples as completely.

### Retrieval-Augmented Generation (RAG)

Decouple factual knowledge from model parameters:

```
ŷ = f_θ(q,  Retrieve(q, K))
```

At inference, relevant document chunks `K` are retrieved and prepended to the context. The model conditions on retrieved content rather than relying solely on what was memorized in `θ`.

This addresses the `Knowledge ⊆ θ` constraint directly — knowledge now lives in an external store, not parameters.

### Larger Synthetic Dataset via Distillation

```
D_synthetic = {(x_i, f_teacher(x_i))}  for  i = 1..N,  N >> 140
```

A larger teacher model (e.g., Mistral-7B) generates responses to diverse prompts. The student (DialoGPT-small) trains on these synthetic pairs. This is the approach behind Stanford Alpaca, Vicuna, and similar open-source instruction models.

### Prefix KV-Cache for Reasoning Traces

As described in the chain-of-thought section, caching key-value states for shared reasoning prefixes reduces inference cost from quadratic to near-linear in the new token count. Implementing this enables practical use of longer, more structured prompts without proportional latency cost.

### Evaluation Metrics (Currently Absent)

| Metric | Measures |
|--------|---------|
| **Perplexity** | How surprised the model is by held-out text — direct measure of fit |
| **BLEU / ROUGE** | N-gram overlap with reference answers |
| **BERTScore** | Semantic similarity via contextual embeddings |
| **Human eval** | Coherence and correctness ratings from annotators |

Without held-out evaluation, there is currently no quantitative signal distinguishing memorization from generalization.

---

## Project Structure

```
logicbot/
│
├── src/
│   ├── train.py          # Fine-tuning pipeline
│   ├── infer.py          # Text generation with logic routing
│   ├── logic_mapper.py   # Keyword-based query classifier
│   └── prepare_data.py   # JSONL loading and tokenization
│
├── data/
│   └── train.jsonl       # Instruction-response pairs
│
├── saved_model/          # Checkpoint output directory
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt

# Train
python src/train.py

# Run inference
python src/infer.py --prompt "What is consciousness?"
```

---

## References

1. Zhang, Y., et al. (2020). *DialoGPT: Large-scale generative pre-training for conversational response generation.* ACL. [[paper]](https://arxiv.org/abs/1911.00536)
2. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI Blog.
3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. [[paper]](https://arxiv.org/abs/1706.03762)
4. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS. [[paper]](https://arxiv.org/abs/2201.11903)
5. Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR. [[paper]](https://arxiv.org/abs/2203.11171)
6. DeepSeek-AI. (2025). *DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning.* [[paper]](https://arxiv.org/abs/2501.12948)
7. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. [[paper]](https://arxiv.org/abs/2106.09685)
8. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [[paper]](https://arxiv.org/abs/2005.11401)
9. Ba, J., et al. (2016). *Layer Normalization.* [[paper]](https://arxiv.org/abs/1607.06450)
10. Pope, R., et al. (2023). *Efficiently Scaling Transformer Inference.* MLSys. [[paper]](https://arxiv.org/abs/2211.05100) — KV-cache analysis
11. HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers

---


## License

MIT License. See `LICENSE` for details.
