# 📚 Obscura: Building an LLM from Scratch

This repository documents the step-by-step journey of building a Large Language Model (LLM), inspired by the groundbreaking paper **“Attention Is All You Need.”**

---

## 📦 Chapter 1: Initial Setup and Dependencies

Set up the development environment and installed essential packages.

### 🔧 Dependencies Installed

* [`uv`](https://github.com/astral-sh/uv): A fast Python package installer and resolver.

---

## ✍️ Chapter 2: Working with Text Data

Begin preparing text data for model training.

### 🧹 Initial Text Processing

* **Data Loading:** Loaded raw text from the *Attention Is All You Need* paper.
* **File Storage:** Saved as `sample_data.txt` for local access.
* **Basic Tokenization:** Used regex to split text (handling spaces, commas, periods).
* **Token Cleaning:** Trimmed whitespace and removed empty tokens.
* **Punctuation Expansion:** Included more punctuation characters in splitting logic.
* **Applied Logic:** Full tokenization of `sample_data.txt`.

### 🧠 Vocabulary and Tokenizers

* **Asterisk Handling:** Adjusted regex to tokenize asterisks correctly.
* **Vocabulary Creation:** Built a unique vocabulary of tokens with size calculation.
* **`SimpleTokenizerV1`:**

  * Converts text ↔️ integer sequences.
  * Maintains `str_to_int` and `int_to_str` mappings.
* **Special Tokens:**

  * `<|endoftext|>`: Marks text/document ends.
  * `<|unk|>`: Represents unknown tokens.
* **`SimpleTokenizerV2`:**

  * Handles out-of-vocabulary tokens by mapping them to `<|unk|>`.
* **Text Concatenation:** Demonstrated usage of `<|endoftext|>` to link multiple text segments.

### 🔄 Data Loading with Tiktoken + PyTorch

* **Tokenizer Upgrade:** Switched to `tiktoken` (used in OpenAI GPT models).
* **`GPTDatasetV1`:**

  * PyTorch `Dataset` using a sliding window.
  * Produces input (context) and target (next token) pairs.
* **DataLoader Utility:** `create_dataloader_v1` helps with batching, shuffling, and multiprocessing.
* **Batch Verification:** Validated the shape and structure of data batches.

### 🧬 Embeddings Implementation

* **Token Embeddings:** Used `torch.nn.Embedding` to convert token IDs → vector representations.
* **Positional Embeddings:** Separate embeddings encode token positions.
* **Combined Embeddings (Planned):** Prepared to merge token and positional embeddings for transformer input.

---

## ⚙️ Chapter 3: Implementing the Attention Mechanism

Delving into the heart of Transformer architecture: **self-attention**.

### 🔁 Core Self-Attention Logic

* **Dot Product Attention:** Manually calculated attention scores via dot products.
* **Softmax:** Applied to scores to obtain probabilistic weights.
* **Context Vector:** Weighted sum of value vectors using attention weights.
* **Matrix Multiplication:** Optimized operations using `@` to parallelize across tokens.

### 🧠 Self-Attention with Trainable Weights

* **Trainable Matrices:** Introduced \$W\_Q\$, \$W\_K\$, \$W\_V\$ for linear projections.
* **Projection Logic:** Transformed embeddings into Q, K, V vectors.
* **Scaled Dot-Product:** Used \$\sqrt{d\_k}\$ to scale dot products and stabilize gradients.
* **`SelfAttention_v1`:** Custom `nn.Module` with manual parameter definitions.
* **`SelfAttention_v2`:** Improved version using `nn.Linear` for weight management.
* **Causal Masking:**

  * Ensures tokens only attend to earlier positions.
  * Implemented using `torch.tril` and `masked_fill_`.

### 🛡️ Regularization with Dropout

* **Dropout Layer:** Added `nn.Dropout` to prevent overfitting.
* **`CausalAttention`:** Integrated QKV projections, dropout, masking into a full attention head.
* **Batch Processing:** Validated compatibility with batched inputs for efficient training.

### 🔍 Multi-Head Attention

* **Conceptual `MultiHeadAttentionWrapper`:**

  * Stacked multiple `CausalAttention` heads.
  * Concatenated their outputs for multi-perspective learning.
* **Efficient `MultiHeadAttention`:**

  * Single QKV projection split into multiple heads using `view` and `transpose`.
  * Final output combined using an `out_proj` linear layer.
* **Demo:** Tested and verified functionality with batch input.
