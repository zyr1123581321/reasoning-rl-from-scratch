from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
tok.pad_token = tok.eos_token

prompts = ["Hello, world!", "This is a test.", "This is another test."]
outputs = ["Hello, world!", "This is a test.", "This is another test."]

# Step 1: tokenize separately, no padding
p_ids = tok(prompts)["input_ids"]
o_ids = tok(outputs)["input_ids"]

for i in range(3):
    print(f"prompt {i}: {p_ids[i]}")
    print(f"output {i}: {o_ids[i]}")
    print(f"combined:  {p_ids[i] + o_ids[i]}")
    print(f"lengths: prompt={len(p_ids[i])}, output={len(o_ids[i])}, combined={len(p_ids[i] + o_ids[i])}")
    print()

# Step 2: concat
combined = [p + o for p, o in zip(p_ids, o_ids)]

# Step 3: pad to max length
max_len = max(len(c) for c in combined)
print(f"max_len: {max_len}")

for c in combined:
    pad_needed = max_len - len(c)
    padded = c + [tok.pad_token_id] * pad_needed
    mask = [1] * len(c) + [0] * pad_needed
    print(f"padded: {padded}")
    print(f"mask:   {mask}")
    print()