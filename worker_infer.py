# worker_infer.py
import json, sys
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

def run(prompt):
    model_name = "h2oai/h2ogpt-4096-llama2-7b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype="auto"
    )
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=800,
        temperature=0.7,
        do_sample=True
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
