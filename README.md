# LogicBot

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Status](https://img.shields.io/badge/Status-Experimental-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An experimental fine-tuning project exploring how small language models behave on custom instruction datasets — and what happens when data is scarce.

---

## What This Project Is

LogicBot is a learning prototype. It fine-tunes **DialoGPT-small** on a custom dataset of logical and philosophical question-answer pairs to explore the mechanics and limits of instruction tuning in low-resource settings.

The project is **not** a production chatbot. It is an honest investigation into what language models can and cannot learn from limited data, and why.

---

## Tools & Libraries Used

| Tool | Purpose |
|------|---------|
| **PyTorch** | Model training, loss computation, backprop |
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

$$P(x_t \mid x_1, x_2, \ldots, x_{t-1}; \theta)$$

Where:
- $x_t$ is the token at position $t$
- $\theta$ represents all learnable parameters (~117M weights)
- The prediction is computed via softmax over the full vocabulary

At inference, this factorization means the model generates one token at a time, left to right.

---

## Transformer Internals

This section explains what happens *inside* each decoder layer — the components that make attention-based models work.

### Scaled Dot-Product Self-Attention

Each token in the sequence attends to every other token by computing three projections from the input embeddings $X$:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

Where:
- $Q$ (Query) — what this token is looking for
- $K$ (Key) — what each token offers to be matched against
- $V$ (Value) — what each token actually contributes to the output
- $d_k$ — key dimension; the $\sqrt{d_k}$ scaling prevents dot products from growing too large, which would push softmax into near-zero gradient regions

The result is a weighted sum of all value vectors, where the weights reflect how relevant each token is to the current one. This is what enables **long-range dependency modeling** — a token at position 50 can directly attend to position 1 with no information bottleneck.

### Multi-Head Attention

Rather than computing one attention map, the model runs $h$ attention heads in parallel:

$$\text{MHA}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W_O$$

$$\text{head}_i = \text{Attention}(XW_{Q_i},\, XW_{K_i},\, XW_{V_i})$$

Each head operates in a lower-dimensional subspace ($d_k = d_{\text{model}} / h$) and learns to attend to different relational patterns simultaneously — syntax, coreference, proximity, topic — which are then concatenated and projected back to model dimension via $W_O$.

DialoGPT-small uses $h = 12$ heads with $d_{\text{model}} = 768$.

### Position-Wise Feedforward Network

After attention, each token's representation passes independently through a two-layer MLP:

$$\text{FFN}(x) = \max(0,\; xW_1 + b_1)\,W_2 + b_2$$

The hidden dimension is typically $4 \times d_{\text{model}}$ (3072 for DialoGPT-small), expanding then compressing the representation. ReLU introduces nonlinearity.

**The critical distinction:** attention *mixes information across tokens* (who attends to whom). FFN *transforms the representation of each token independently* (what does that mixed information mean). Both are necessary — attention alone is a linear operation.

### Residual Connections and Layer Normalization

Each sublayer (attention and FFN) is wrapped with a residual connection:

$$x' = x + \text{Sublayer}(x)$$

Followed by Layer Normalization:

$$\text{LN}(x) = \frac{x - \mu}{\sigma} \cdot \gamma + \beta$$

Where $\mu$ and $\sigma$ are the mean and standard deviation computed across the feature dimension, and $\gamma$, $\beta$ are learnable scale and shift parameters.

**Why this matters for training:** residual connections create a gradient highway — the loss gradient can flow directly back to early layers without passing through many nonlinearities. Without them, 12-layer transformers suffer from vanishing gradients and fail to train. LayerNorm keeps activations in a stable range across layers, preventing representation drift as depth increases.

---

## Reasoning and Chain-of-Thought

### What "Reasoning" Actually Means in Autoregressive Models

This is an important distinction that is frequently misunderstood.

Language models do **not** perform symbolic reasoning. They do not maintain a logic engine, a proof state, or a working memory. Every output token is computed the same way — via attention over the context window followed by a softmax over the vocabulary.

What is called "reasoning" is more precisely: **conditional token generation that statistically mimics reasoning traces seen during training.**

The base probability factorization remains unchanged:

$$P(y \mid x) = \prod_{t=1}^{T} P(y_t \mid y_{<t}, x;\, \theta)$$

### Chain-of-Thought as Latent Trajectory Sampling

When a model is prompted to produce intermediate steps before an answer, it generates a reasoning trace $r_1, r_2, \ldots, r_n$ as part of the output sequence:

$$x \;\longrightarrow\; r_1 \;\longrightarrow\; r_2 \;\longrightarrow\; \cdots \;\longrightarrow\; y$$

This can be written as marginalizing over latent reasoning paths:

$$P(y \mid x) \approx \sum_{r} P(y \mid r, x)\, P(r \mid x)$$

Each intermediate token $r_i$ conditions subsequent generation, effectively decomposing a hard prediction into smaller steps. This works because:

1. The training data contains explanations, proofs, and step-by-step solutions
2. Each reasoning step narrows the probability distribution over the final answer
3. Composing shorter conditional predictions is easier than a single long one

**The honest framing:** chain-of-thought is latent trajectory sampling in token space. Reasoning "emerges" when the training distribution contains enough structured explanation that intermediate tokens reliably predict correct final tokens. It is a statistical phenomenon, not a cognitive one.

LogicBot's training data does not contain reasoning traces, so chain-of-thought behavior is not expected to emerge here.

### Self-Reflection and Iterative Refinement

Some systems implement iterative generation, where the model critiques and revises its own output:

$$y_0 = f_\theta(x)$$
$$y_{t+1} = f_\theta(x,\; y_t)$$

Each pass uses the previous output as additional context. In fixed-point terms, this searches for:

$$y^* = f_\theta(x,\; y^*)$$

This approximates deliberative reasoning by giving the model multiple passes over its own output. It is used in self-consistency decoding (sampling multiple reasoning paths and taking a majority vote) and in Constitutional AI-style revision pipelines. It is **not** implemented in LogicBot, but represents a concrete direction for v2.

---

## Training Method

Fine-tuning adapts the pretrained weights toward a new task using supervised learning on labeled examples.

**Input format:**
```
Question: <prompt>
Answer: <response>
```

**Objective — Cross-Entropy Loss:**

$$\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log P(y_t \mid y_{<t}, x; \theta)$$

Where:
- $x$ = input prompt tokens
- $y$ = target response tokens
- $T$ = sequence length
- The model is penalized proportionally to how surprised it is by each correct token

The gradient of this loss is backpropagated through all transformer layers to update $\theta$.

**Why this works (in theory):** By minimizing cross-entropy over many examples, the model adjusts its internal representations to favor the patterns present in the training data.

**Why this has limits here:** With ~140 examples, those patterns are too sparse to generalize. The model fits the training distribution, not the underlying task.

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

The relationship between dataset size and model behavior is well-studied:

| Dataset Size | Expected Behavior |
|--------------|------------------|
| < 500 samples | Memorization of surface patterns |
| 1k – 5k | Early pattern learning, limited generalization |
| 10k+ | Style-level generalization begins |
| 100k+ | Semantic reasoning starts to emerge |

Formally, the current dataset $\mathcal{D}$ satisfies:

$$|\mathcal{D}| \ll |\mathcal{D}_{\text{critical}}|, \quad |\mathcal{D}_{\text{critical}}| \approx 10^4$$

This means the model cannot learn abstract relationships — it learns which words tend to follow which other words in the training examples.

---

## Training Configuration

```python
EPOCHS     = 10
BATCH_SIZE = 2
LR         = 2e-5
MAX_LEN    = 256
OPTIMIZER  = AdamW(weight_decay=0.01)
```

**AdamW** extends Adam with decoupled weight decay:

$$\theta_{t+1} = \theta_t - \alpha \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t \right)$$

Where $\hat{m}_t$ and $\hat{v}_t$ are bias-corrected first and second moment estimates of the gradient, and $\lambda$ is the weight decay coefficient. Weight decay penalizes large parameter values, acting as L2 regularization to reduce overfitting.

---

## Inference

Text generation uses **nucleus (top-p) sampling**:

$$P_{\text{sample}}(x_t) \propto P(x_t) \cdot \mathbf{1}\left[x_t \in \text{TopP}(p)\right]$$

The model only samples from the smallest set of tokens whose cumulative probability exceeds $p$.

```python
temperature = 0.6   # Sharpens the distribution (lower = less random)
top_p       = 0.9   # Truncates low-probability tail
```

**Effect of temperature** on the probability distribution:

$$P_{\text{scaled}}(x) = \frac{\exp(\log P(x) / T)}{\sum_{x'} \exp(\log P(x') / T)}$$

- $T < 1$: Distribution sharpened, model favors high-probability tokens
- $T > 1$: Distribution flattened, more uniform sampling
- $T \to 0$: Equivalent to greedy decoding ($\arg\max$)

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

This is a **heuristic keyword matcher** — it does not perform symbolic reasoning or semantic parsing. It adjusts the prompt prefix passed to the model, which weakly guides output style. The model itself has no reasoning module.

---

## Why the Model Underperforms

These are the concrete, technical reasons — not excuses.

### 1. Data scarcity (primary cause)

140 samples cannot cover the input distribution the model will encounter at inference. The model overfits:

$$\mathcal{L}_{\text{train}} \downarrow \quad \text{while} \quad \mathcal{L}_{\text{test}} \uparrow$$

The parameters converge to memorized training fragments, not generalizable representations.

### 2. No instruction pretraining

DialoGPT was pretrained on casual conversation, not instruction-following. It has no internal structure for task-response alignment. Fine-tuning on 140 examples cannot overcome a mismatched pretraining distribution.

### 3. No RLHF or preference optimization

The training signal is next-token prediction loss only. There is no:
- Reward model
- Human preference data
- KL penalty to keep outputs stable

Outputs are unfiltered by quality or coherence.

### 4. No retrieval component

All knowledge must be encoded in the model weights $\theta$. There is no external memory:

$$\text{Knowledge} \subseteq \theta$$

The model cannot look anything up. Factual accuracy is entirely constrained by what the base model learned during pretraining, partially overwritten by fine-tuning.

### 5. Known failure modes

Due to the above, the model exhibits documented autoregressive failure patterns:

| Failure Mode | Cause |
|---|---|
| Token repetition (`computable.computable...`) | Mode collapse in low-entropy states |
| Semantic drift | Attention patterns not anchored to prompt |
| Hallucination | No factual grounding, confident sampling |
| Incoherent definitions | Vocabulary from training, logic absent |

These are expected behaviors in undertrained small models — not bugs.

---

## What This Project Does Demonstrate

Despite the limitations, building this end-to-end covers real ML engineering:

- Tokenization pipeline and padding strategy
- JSONL dataset loading and preprocessing
- HuggingFace `Trainer` configuration
- Gradient accumulation with small batch sizes
- Nucleus sampling inference
- Prompt engineering for instruction-style models
- Quantitative error analysis

Understanding *why* a model fails is as technically valuable as making one succeed.

---

## Planned Improvements

### LoRA (Low-Rank Adaptation)

Instead of updating all 117M parameters, LoRA injects trainable rank-decomposition matrices:

$$W' = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d,k)$$

This reduces trainable parameters by 10–100×, enabling fine-tuning on limited hardware without overfitting large weight matrices.

### Retrieval-Augmented Generation (RAG)

Decouple factual knowledge from model parameters:

$$\hat{y} = f_\theta\left(q,\ \text{Retrieve}(q, \mathcal{K})\right)$$

The model conditions on retrieved document chunks $\mathcal{K}$ at inference time, removing the constraint that all knowledge must live in weights.

### Larger Synthetic Dataset

Generate 10k+ instruction pairs using a larger model as a teacher, then fine-tune on the synthetic data. This is the approach used by Alpaca, Vicuna, and similar projects.

### Evaluation Metrics

Currently no quantitative evaluation. Planned additions:

| Metric | Measures |
|--------|---------|
| **Perplexity** | How surprised the model is by held-out text |
| **BLEU / ROUGE** | N-gram overlap with reference answers |
| **BERTScore** | Semantic similarity via embeddings |
| **Human eval** | Coherence and correctness ratings |

---

## Project Structure

```
logicbot/
│
├── train.py          # Fine-tuning pipeline
├── inference.py      # Text generation with logic routing
├── dataset.py        # JSONL loading and tokenization
├── logic_modes.py    # Keyword-based query classifier
├── data/
│   └── train.jsonl   # Instruction-response pairs
├── saved_model/      # Checkpoint output directory
├── requirements.txt
└── README.md
```

---

## Setup

```bash
pip install -r requirements.txt

# Train
python train.py

# Run inference
python inference.py --prompt "What is entropy?"
```

---

## References

1. Zhang, Y., et al. (2020). *DialoGPT: Large-scale generative pre-training for conversational response generation.* ACL. [[paper]](https://arxiv.org/abs/1911.00536)
2. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI.
3. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. [[paper]](https://arxiv.org/abs/1706.03762) — Original Transformer architecture
4. Wei, J., et al. (2022). *Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.* NeurIPS. [[paper]](https://arxiv.org/abs/2201.11903)
5. Wang, X., et al. (2023). *Self-Consistency Improves Chain of Thought Reasoning in Language Models.* ICLR. [[paper]](https://arxiv.org/abs/2203.11171)
6. Hu, E., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. [[paper]](https://arxiv.org/abs/2106.09685)
7. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS. [[paper]](https://arxiv.org/abs/2005.11401)
8. Ba, J., et al. (2016). *Layer Normalization.* [[paper]](https://arxiv.org/abs/1607.06450)
9. HuggingFace Transformers Documentation: https://huggingface.co/docs/transformers

---

## License

MIT License. See `LICENSE` for details.