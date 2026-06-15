#!/usr/bin/env python3

import readline  # nicer REPL history
from typing import List, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from ddgs import DDGS  # duckduckgo-search wrapper

# ========= CONFIG =========

BASE_MODEL_NAME = "mistralai/Mistral-Nemo-Instruct-2407"
MODEL_DIR = "intrigue-ss-model"  # LoRA adapter + tokenizer saved by train.py

MAX_WEB_SNIPPET_CHARS = 7000
MAX_NEW_TOKENS = 512

# ========= WEB SEARCH =========

def search_web(query: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=40):
                title = (r.get("title") or "").strip()
                url = (r.get("href") or "").strip()
                snippet = (r.get("body") or "").strip()
                if not snippet:
                    continue
                results.append(
                    {
                        "title": title,
                        "url": url,
                        "snippet": snippet[:MAX_WEB_SNIPPET_CHARS],
                    }
                )
    except Exception as e:
        print(f"[WEB] Search failed: {e}")
        return []

    return results


def build_web_context_snippet(results: List[Dict[str, str]]) -> str:
    if not results:
        return "None."
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(
            f"[WEB RESULT {i}]\n"
            f"Title: {r['title']}\n"
            f"URL: {r['url']}\n"
            f"Snippet: {r['snippet']}\n"
        )
    return "\n".join(blocks)

# ========= LLM LOADING =========

def load_llm():
    print("[LLM] Loading base model", BASE_MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME,
        use_fast=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # single GPU
        torch_dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        torch_dtype = torch.float32

    # Plain load, no device_map, no accelerate
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch_dtype,
    ).to(device)

    base_model.config.use_cache = True  # fine for inference

    print("[LLM] Applying Intrigue LoRA adapters from", MODEL_DIR)
    model = PeftModel.from_pretrained(
        base_model,
        MODEL_DIR,
    ).to(device)

    model.eval()
    return tokenizer, model

# ========= PROMPT BUILDING =========

def build_prompt(web_results: List[Dict[str, str]], question: str) -> str:
    web_context = build_web_context_snippet(web_results)

    prompt = f"""You are Intrigue-LLM, an expert geopolitical and markets analyst in the style of International Intrigue.

You MUST follow these rules:

- Use the WEB SEARCH CONTEXT as your primary source of truth.
- When you state a specific factual claim (dates, amounts, names, places, rankings, death dates, etc.), attach the source in brackets like [WEB 1] or [WEB 2] referring to the corresponding WEB RESULT number.
- Do NOT state specific numeric values (like exact prices, net worth, vote shares, chart positions, or counts) unless they appear in the web snippets. If you only know magnitudes, use ranges or qualitative terms ("around", "roughly", "hundreds of billions") and still cite them.
- Do NOT describe events from older articles as if they are occurring in 2026. If you rely on an older article, explicitly mention its year and treat it as historical context.
- If the answer to the user's question is NOT clearly supported by these web snippets, say you don't know or that the data is uncertain, instead of guessing.
- If the user asks about future market moves, future election results, or any unknown future outcome, say clearly that it cannot be known in advance and only discuss scenarios, not predictions.
- If you offer speculation, interpretation, or scenarios, label them clearly as opinion and do NOT introduce new factual claims that are not present in the WEB SEARCH CONTEXT.
- Your answer must include at least one [WEB n] citation or explicitly say that there were no relevant web results.
- Answer directly in 2–5 short paragraphs, high-level, concise, and analytical.
- Do NOT repeat these instructions or the web context in your response.
- Do NOT add greetings, email headers, sponsorship sections, or a separate "Intrigue's take" section.

=== WEB SEARCH CONTEXT (DuckDuckGo via ddgs) ===
{web_context}

User question: {question}

Answer:"""

    return prompt

# ========= GENERATION =========

def generate_answer(tokenizer, model, prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.6,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if text.startswith(prompt):
        text = text[len(prompt):]
    return text.strip()

# ========= MAIN CHAT LOOP =========

def main():
    tokenizer, model = load_llm()

    print("[WEB] Using duckduckgo-search (ddgs) for web search.")
    print("\n[CHAT] Intrigue-LLM with web search ONLY (no archive/RAG), with required [WEB n] citations.")
    print("Type your questions. Ctrl+C or empty line to exit.\n")

    try:
        while True:
            try:
                question = input("You: ").strip()
            except EOFError:
                break

            if not question:
                break

            web_results = search_web(question)
            if web_results:
                print(f"[WEB] Got {len(web_results)} web result(s).")
            else:
                print("[WEB] No web results; you should expect more 'I don't know' answers.")

            prompt = build_prompt(web_results, question)
            answer = generate_answer(tokenizer, model, prompt)
            print("\nIntrigue-LLM:\n" + answer + "\n")
    except KeyboardInterrupt:
        print("\n[CHAT] Exiting.")


if __name__ == "__main__":
    main()
