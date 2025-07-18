
# 📚 Obscura: Building an LLM from Scratch

This repository documents the step-by-step journey of building a Large Language Model (LLM), inspired by the groundbreaking paper *“Attention Is All You Need.”*

---

## 📦 Chapter 1: Initial Setup and Dependencies

Set up the development environment and installed essential packages.

### 🔧 Dependencies Installed

* `uv`: A fast Python package installer and resolver.

---

## ✍️ Chapter 2: Working with Text Data

Begin preparing text data for model training.

### 🧹 Initial Text Processing

* **Data Loading**: Loaded raw text from the *Attention Is All You Need* paper.
* **File Storage**: Saved as `sample_data.txt` for local access.
* **Basic Tokenization**: Used regex to split text (handling spaces, commas, periods).
* **Token Cleaning**: Trimmed whitespace and removed empty tokens.
* **Punctuation Expansion**: Included more punctuation characters in splitting logic.
* **Applied Logic**: Full tokenization of `sample_data.txt`.

### 🧠 Vocabulary and Tokenizers

* **Asterisk Handling**: Adjusted regex to tokenize asterisks correctly.
* **Vocabulary Creation**: Built a unique vocabulary of tokens with size calculation.

#### SimpleTokenizerV1

* Converts text ↔️ integer sequences.
* Maintains `str_to_int` and `int_to_str` mappings.

**Special Tokens**:

* `<|endoftext|>`: Marks text/document ends.
* `<|unk|>`: Represents unknown tokens.

#### SimpleTokenizerV2

* Handles out-of-vocabulary tokens by mapping them to `<|unk|>`.
* **Text Concatenation**: Demonstrated usage of `<|endoftext|>` to link multiple text segments.

---

## 🔄 Data Loading with Tiktoken + PyTorch

### Tokenizer Upgrade

* Switched to `tiktoken` (used in OpenAI GPT models).

### GPTDatasetV1

* PyTorch Dataset using a sliding window.
* Produces input (context) and target (next token) pairs.

### DataLoader Utility

* `create_dataloader_v1`: Helps with batching, shuffling, and multiprocessing.
* **Batch Verification**: Validated the shape and structure of data batches.

---

## 🧬 Embeddings Implementation

* **Token Embeddings**: Used `torch.nn.Embedding` layer to convert token IDs → vector representations.
* **Positional Embeddings**: Separate embeddings encode token positions.
* **Combined Embeddings (Conceptual)**: Prepared to merge token and positional embeddings for transformer input.

---

## ⚙️ Chapter 3: Implementing the Attention Mechanism

Delving into the heart of Transformer architecture: self-attention.

### 🔁 Core Self-Attention Logic

* **Dot Product Attention**: Manually calculated attention scores via dot products.
* **Softmax**: Applied to scores to obtain probabilistic weights.
* **Context Vector**: Weighted sum of value vectors using attention weights.
* **Matrix Multiplication**: Optimized operations using `@` to parallelize across tokens.

### 🧠 Self-Attention with Trainable Weights

* **Trainable Matrices**: Introduced `W_Q`, `W_K`, `W_V` for linear projections.
* **Projection Logic**: Transformed embeddings into Q, K, V vectors.
* **Scaled Dot-Product**: Used `sqrt(d_k)` to scale dot products and stabilize gradients.

#### SelfAttention\_v1

* Custom `nn.Module` with manual parameter definitions.

#### SelfAttention\_v2

* Improved version using `nn.Linear` for weight management.

### 🛡️ Causal Masking

* Ensures tokens only attend to earlier positions.
* Implemented using `torch.tril` and `masked_fill_`.

### 🧯 Regularization with Dropout

* **Dropout Layer**: Added `nn.Dropout` to prevent overfitting.
* **CausalAttention**: Integrated QKV projections, scaled dot-product attention, causal masking, and dropout into a single attention head.
* **Batch Processing**: Validated compatibility with batched inputs.

---

## 🔍 Multi-Head Attention

### Conceptual `MultiHeadAttentionWrapper`

* Stacked multiple `CausalAttention` heads.
* Concatenated outputs for multi-perspective learning.

### Efficient `MultiHeadAttention`

* Single QKV projection split into multiple heads using `view` and `transpose`.
* Final output combined using an `out_proj` linear layer.
* **Demo**: Tested and verified functionality with batch input.

---

## 🏗️ Chapter 4: Coding an LLM Architecture

This chapter brings all the components together to build the full GPT (Generative Pre-trained Transformer) model.

### GPT Model Structure (Dummy Implementation)

#### `GPT_CONFIG_124M`

* Configuration dictionary for a GPT-like model with 124M parameters (GPT-2 Small).
* Specifies: `vocab_size`, `context_length`, `emb_dim`, `n_heads`, `n_layers`, `drop_rate`, `qkv_bias`.

#### DummyGPTModel

* Token embeddings
* Positional embeddings
* Dropout
* Sequence of Transformer blocks
* Final layer normalization
* Output head

> Uses `DummyTransformerBlock` and `DummyLayerNorm` that pass inputs directly.

#### Model Instantiation and Forward Pass

* Demonstrated model instantiation and forward pass with sample inputs.

---

### 🧪 Layer Normalization and GELU Activation

#### Layer Normalization

* Manually calculated mean/variance across feature dimension.
* Implemented custom `LayerNorm` with learnable scale/shift, epsilon for stability.

#### GELU Activation Function

* Introduced and implemented custom GELU function.
* Compared it with ReLU using visualizations.

---

### ⚙️ FeedForward Network

* Two linear layers with GELU in between.
* Processes each token's embedding independently.

---

### 🔗 Residual Connections (Shortcut Connections)

#### Vanishing Gradients

* Demonstrated how gradients vanish in deep nets without shortcut connections.

#### Impact of Shortcut Connections

* Showed improvement in gradient flow using residual connections.

#### Conceptual Integration

* Explained how residual connections apply in Transformer blocks.

---

## 🧱 Transformer Block Implementation

### TransformerBlock Class

* Multi-Head Attention
* Feed-Forward Network
* Layer Normalization (pre-attention and pre-FFN)
* Residual Connections
* Dropout

> **Block Integration Test**: Verified shape consistency through the block.

---

## 🧠 Complete GPT Model and Text Generation

### GPTModel Class

* `nn.Embedding` layers for tokens and positions
* Dropout
* Stack of `TransformerBlock`s
* Final `LayerNorm`
* Output `Linear` head (for vocab prediction)

### Parameter Counting

* Printed total number of parameters (e.g., 124M for GPT-2 Small).
* Explained weight tying to reduce parameters.
* Estimated memory footprint.

### `generate_text_simple` Function

* Basic greedy decoding function.
* Crops context to `context_length`.
* Demonstrates full input → output flow of the model.
