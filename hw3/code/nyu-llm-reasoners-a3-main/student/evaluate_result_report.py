import json


print("==================Intellect Results===================\n")
with open("eval_intellect.json") as f:
    data = json.load(f)
    results = data["results"]

both_correct = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0)
format_correct = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] != 1.0)
not_correct = sum(1 for r in results if r["format_reward"] != 1.0 and r["answer_reward"] != 1.0)
fmt_zero = [r for r in data["results"] if r["format_reward"] == 0.0]
fmt_one_answer_zero = [r for r in data["results"] if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]

print(f"Both correct:       {both_correct}")
print(f"Format correct:     {format_correct}")
print(f"Not correct:        {not_correct}")
print(f"Total:              {len(results)}")

print(f"=== FORMAT = 0 ({len(fmt_zero)} total, showing 10) ===\n")
for r in fmt_zero[:10]:
    print(f'=== index={r["idx"]} | GT: {r["ground_truth"]} ===')
    print(r["response"][-300:])
    print()

print(f"=== FORMAT = 1.0, ANSWER = 0.0 ({len(fmt_one_answer_zero)} total, showing 10) ===\n")
for r in fmt_one_answer_zero[:10]:
    resp = r["response"]
    idx = resp.rfind(r"\boxed{")
    boxed = resp[idx:idx+60] if idx != -1 else "NOT FOUND"
    print(f'=== index={r["idx"]} | GT: {r["ground_truth"]} === | predicted {boxed}')
    print(resp[-300:])
    print()




print("==================MATH Results===================\n")
with open("eval_math.json") as f:
    data = json.load(f)
    results = data["results"]

both_correct = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0)
format_correct = sum(1 for r in results if r["format_reward"] == 1.0 and r["answer_reward"] != 1.0)
not_correct = sum(1 for r in results if r["format_reward"] != 1.0 and r["answer_reward"] != 1.0)
fmt_zero = [r for r in data["results"] if r["format_reward"] == 0.0]
fmt_one_answer_zero = [r for r in data["results"] if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]


print(f"Both correct:       {both_correct}")
print(f"Format correct:     {format_correct}")
print(f"Not correct:        {not_correct}")
print(f"Total:              {len(results)}")

print(f"=== FORMAT = 0 ({len(fmt_zero)} total, showing 10) ===\n")
for r in fmt_zero[:10]:
    print(f'=== index={r["idx"]} | GT: {r["ground_truth"]} ===')
    print(r["response"][-300:])
    print()

print(f"=== FORMAT = 1.0, ANSWER = 0.0 ({len(fmt_one_answer_zero)} total, showing 10) ===\n")
for r in fmt_one_answer_zero[:10]:
    resp = r["response"]
    idx = resp.rfind(r"\boxed{")
    boxed = resp[idx:idx+60] if idx != -1 else "NOT FOUND"
    print(f'=== index={r["idx"]} | GT: {r["ground_truth"]} === | predicted {boxed}')
    print(resp[-300:])
    print()