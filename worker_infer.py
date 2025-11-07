# worker_infer.py
import json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

def run(prompt):
    # ⚙️ Use a smaller and faster model
    model_name = "h2oai/h2ogpt-4096-llama2-1.3b"  # about 6× smaller than 7B

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=600,      # shorter output -> less memory
        temperature=0.7,
        do_sample=True
        device=-1
    )

    out = generator(prompt, num_return_sequences=1)
    text = out[0]["generated_text"]
    print(json.dumps({"ok": True, "text": text}))

if __name__ == "__main__":
    try:
        prompt = sys.stdin.read()
        run(prompt)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}))
        sys.exit(1)
