#@title LoRA fine-tune one agent on Colab (T4 GPU) and register it in Ollama
"""Train a LoRA adapter on a dataset exported from the "Fine-tune agents" page, then make it an Ollama model.

    !python deploy/colab/finetune_agent.py --data understanding_train.jsonl --base qwen2.5-coder:3b --name understanding-ft

Steps: install Unsloth -> load the base model in 4 bit -> LoRA SFT on the chat examples -> export GGUF (q4_k_m)
-> copy the base model's Ollama chat template -> `ollama create <name>`. Then enter <name> for that agent on the
page (section c). Runs on the Colab host from all_in_one.py, where Ollama is already running.
"""
import argparse
import glob
import json
import os
import subprocess
import sys

# Ollama tag -> Hugging Face weights of the same model (Unsloth builds, pre-quantised for 4-bit training)
BASES = {
    "qwen2.5-coder:3b": "unsloth/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-coder:7b": "unsloth/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5:3b": "unsloth/Qwen2.5-3B-Instruct",
    "qwen2.5:7b-instruct": "unsloth/Qwen2.5-7B-Instruct",
    "qwen3:4b": "unsloth/Qwen3-4B",
    "qwen3:8b": "unsloth/Qwen3-8B",
}


def sh(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"command failed: {cmd}\n{(r.stderr or r.stdout)[-1500:]}")
    return r.stdout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="chat JSONL exported from the Fine-tune agents page")
    ap.add_argument("--base", required=True, help="Ollama tag of the model the agent uses now, e.g. qwen2.5-coder:3b")
    ap.add_argument("--name", required=True, help="name of the new Ollama model, e.g. understanding-ft")
    ap.add_argument("--hf", default="", help="Hugging Face id of the base weights (default: looked up from --base)")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--out", default="/content/finetuned")
    a = ap.parse_args()

    hf_id = a.hf or BASES.get(a.base)
    if not hf_id:
        sys.exit(f"Unknown base '{a.base}'. Pass --hf <huggingface id>. Known: {', '.join(BASES)}")
    rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]
    print(f"OK   {len(rows)} examples from {a.data}")
    if len(rows) < 20:
        print("WARN fewer than 20 examples: the adapter will barely change the model")

    print("...  installing Unsloth (2-4 min)")
    sh(f"{sys.executable} -m pip install -q unsloth")
    from unsloth import FastLanguageModel   # import first: Unsloth patches transformers / trl
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    model, tok = FastLanguageModel.from_pretrained(hf_id, max_seq_length=a.max_len, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=a.rank, lora_alpha=a.rank, lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    data = Dataset.from_list([{"text": tok.apply_chat_template(r["messages"], tokenize=False)} for r in rows])

    trainer = SFTTrainer(model=model, tokenizer=tok, train_dataset=data, args=SFTConfig(
        dataset_text_field="text", max_seq_length=a.max_len, per_device_train_batch_size=2,
        gradient_accumulation_steps=4, num_train_epochs=a.epochs, learning_rate=a.lr, warmup_steps=5,
        logging_steps=5, optim="adamw_8bit", lr_scheduler_type="linear", seed=42, report_to="none",
        output_dir=os.path.join(a.out, "checkpoints")))
    stats = trainer.train()
    print(f"OK   trained in {stats.metrics.get('train_runtime', 0):.0f}s, final loss {stats.metrics.get('train_loss', 0):.3f}")

    gguf_dir = os.path.join(a.out, a.name)
    model.save_pretrained_gguf(gguf_dir, tok, quantization_method="q4_k_m")
    ggufs = sorted(glob.glob(os.path.join(gguf_dir, "**", "*.gguf"), recursive=True), key=os.path.getsize)
    if not ggufs:
        sys.exit(f"FAIL no .gguf written in {gguf_dir}")
    print(f"OK   GGUF {ggufs[-1]}")

    # same family as the base: reuse its chat template and parameters, only the weights change
    sh(f"ollama pull {a.base}")
    base_file = sh(f"ollama show --modelfile {a.base}")
    lines = [l for l in base_file.splitlines() if not l.startswith("FROM ") and not l.startswith("#")]
    modelfile = os.path.join(gguf_dir, "Modelfile")
    with open(modelfile, "w", encoding="utf-8") as f:
        f.write(f"FROM {ggufs[-1]}\n" + "\n".join(lines) + "\n")
    sh(f"ollama create {a.name} -f {modelfile}")
    print(f"OK   Ollama model '{a.name}' created. On the Fine-tune agents page, enter it under 'c. Model for this agent'.")


if __name__ == "__main__":
    main()
