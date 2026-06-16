# Intrigue NeMo LoRA

This is a small setup for training and running a custom **Intrigue-style** large language model on your own data using **Mistral-Nemo-Instruct-2407** with **LoRA** and web search.

It has two main pieces:

- `train.py` – fine-tunes NeMo 12B with LoRA on the Intrigue newsletter corpus.
- `chat.py` – runs an interactive CLI bot that does web search + Intrigue-style analysis using the trained adapter.
- `scrape.py` - scrapes all of the newsletters from the International Intrigue for train.py to make an index

---

## 1. Environment and dependencies

Install the dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Key libraries:

- `torch` – PyTorch
- `transformers` – Hugging Face models, tokenizer, Trainer
- `datasets` – dataset handling and tokenization
- `peft` – LoRA (parameter-efficient fine-tuning)
- `accelerate` – multi-GPU training (2×A40 used here)
- `duckduckgo-search` – web search via DDGS
- `tqdm and cloudscraper` – progress bars and bypassing cloudflare

You also need a working CUDA setup (NVIDIA A40s in this case) and an HF token if you want faster/more reliable downloads from the Hub.

---

## 2. Data layout

Expected directory structure:

```text
~/data/
  intrigue_clean/
    train.py          # LoRA training script
    chat.py           # CLI web-search chat bot
    intrigue-ss-model # output adapter dir (created by train.py)
    *.txt             # newsletter files for training
```

Each `.txt` file in `intrigue_clean` is treated as one newsletter. The first three non-empty lines are dropped as boilerplate; the rest is used as training text.

---

## 3. Training the LoRA (NeMo 12B + Accelerate)

`train.py` fine-tunes `mistralai/Mistral-Nemo-Instruct-2407` using LoRA on your newsletter corpus. It:

- Loads all `.txt` files from `intrigue_clean/`.
- Concatenates and chunks into sequences (default 512 tokens).
- Tokenizes with the NeMo tokenizer (`fix_mistral_regex=True`).
- Loads the base NeMo 12B model in bf16.
- Wraps it with a LoRA adapter (rank 16) using PEFT.
- Uses Hugging Face `Trainer` + Accelerate to train on **2 GPUs**.

### Configure Accelerate (one-time) if you have a large machine

From your shell (not inside Python):

```bash
accelerate config
```

Use:

- Environment: `This machine`
- Machine type: `multi-GPU`
- GPUs: `2`, IDs `0,1`
- Mixed precision: `bf16`
- No DeepSpeed / FSDP / Megatron

This writes `~/.cache/huggingface/accelerate/default_config.yaml`.

### Run training

From `~/data/intrigue_clean`:

```bash
screen -S intrigue-train
cd ~/data/intrigue_clean

export TOKENIZERS_PARALLELISM=false

accelerate launch train.py
```

Training details (as configured):

- Base model: `mistralai/Mistral-Nemo-Instruct-2407`
- Context length: 512 tokens
- Batch: per-device batch size 1, gradient accumulation 4
- Epochs: 5
- LoRA: rank 16, alpha 32, dropout 0.05
- Precision: bf16 on GPU

Output:

- LoRA adapter and tokenizer are saved into `intrigue-ss-model/` via `model.save_pretrained()` and `tokenizer.save_pretrained()`.

---

## 4. Running the Intrigue chat bot

`chat.py` runs a REPL that:

- Loads the NeMo base model.
- Applies the Intrigue LoRA adapter from `intrigue-ss-model/`.
- Uses DuckDuckGo (DDGS) to fetch live web results.
- Builds a prompt that forces the model to:
  - Use the web snippets as primary evidence.
  - Cite sources as `[WEB n]`.
  - Avoid overconfident speculation and fake numbers when not present in the snippets.
- Answers in a concise, analytic “International Intrigue” style.

### Usage

Make sure **no training job is still running** and GPUs have free memory. Then:

```bash
cd ~/data/intrigue_clean
python3 chat.py
```

You’ll see:

```text
[LLM] Loading base model mistralai/Mistral-Nemo-Instruct-2407
[LLM] Applying Intrigue LoRA adapters from intrigue-ss-model
[WEB] Using duckduckgo-search (ddgs) for web search.

[CHAT] Intrigue-LLM with web search ONLY (no archive/RAG), with required [WEB n] citations.
Type your questions. Ctrl+C or empty line to exit.
```

Then just type:

```text
You: what's going on with SPCX today
You: what would Intrigue say about Hakeem Butler to the Broncos
You: is the A40 actually a good GPU for local LLM inference
```

The bot will:

- Run a DuckDuckGo search.
- Build a numbered `WEB RESULT n` context block.
- Generate an answer with inline `[WEB n]` citations.

---

## 5. Storage and cache notes

Hugging Face caches model weights under:

```text
~/.cache/huggingface/hub/
```

The LoRA adapter and tokenizer you care about live in:

```text
intrigue-ss-model/
```

Do **not** delete that if you want to keep using the Intrigue model.

---

## 6. Quick troubleshooting

- **CUDA OOM on training**: reduce `MAX_SEQ_LEN`, `GRAD_ACCUM_STEPS`, or epochs in `train.py`, and make sure no other heavy jobs are on the GPUs.
- **“You shouldn’t move a model that is dispatched using accelerate hooks”** when running `chat.py`: don’t use `accelerate launch` for `chat.py` and don’t use `device_map` there; `chat.py` should be a simple single-GPU load.
- **Tokenizers thread-pool panic**: ensure `TOKENIZERS_PARALLELISM=false` is set in the environment and at the top of `train.py`.

---
