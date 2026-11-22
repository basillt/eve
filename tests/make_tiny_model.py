"""
Build a tiny random MoE model + tokenizer locally, so the whole pipeline can be
tested without downloading anything. Results from it are meaningless — it only
checks that the code runs.

    python tests/make_tiny_model.py --arch granitemoe --out tests/tiny-granitemoe
"""

import argparse
import glob
import json

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

CHAT_TEMPLATE = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
                 "{% if add_generation_prompt %}<|assistant|>{% endif %}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="granitemoe", choices=["granitemoe", "olmoe", "mixtral", "qwen2_moe", "qwen2", "llama"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    texts = []
    for path in glob.glob("prompts/*.jsonl"):
        with open(path, encoding="utf-8") as f:
            texts += [json.loads(l)["text"] for l in f if l.strip()]

    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(texts, trainers.BpeTrainer(
        vocab_size=1000, special_tokens=["<unk>", "<pad>", "<eos>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")
    tok.chat_template = CHAT_TEMPLATE

    common = dict(vocab_size=len(tok), hidden_size=64, intermediate_size=64, num_hidden_layers=4,
                  num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=1024,
                  num_experts_per_tok=4, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    if args.arch == "qwen2_moe":
        extra = dict(num_experts=16, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    elif args.arch in ("qwen2", "llama"):
        extra = dict(intermediate_size=256)
        common.pop("num_experts_per_tok")
        common.pop("intermediate_size")
    elif args.arch == "olmoe":
        extra = dict(num_experts=16)
    else:
        extra = dict(num_local_experts=16)
    config = AutoConfig.for_model(args.arch, **common, **extra)
    model = AutoModelForCausalLM.from_config(config)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved tiny {args.arch} to {args.out}")


if __name__ == "__main__":
    main()
