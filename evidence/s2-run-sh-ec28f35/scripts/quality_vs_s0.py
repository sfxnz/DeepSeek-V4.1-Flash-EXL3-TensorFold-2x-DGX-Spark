"""s2: ctx1m/quality_quick.json against the s0 receipts' quick run (TensorFold 0d91389, --context 65538,
data/tf-dsv41/receipts/final/quality_quick.json), field by field, time fields left out.   python3 scripts/quality_vs_s0.py"""
import json

S0 = "/home/sfxnz/projects/data/tf-dsv41/receipts/final/quality_quick.json"
a, b = json.load(open("ctx1m/quality_quick.json"))["components"], json.load(open(S0))["components"]


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k != "s"}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


print(f"s2: ctx1m/quality_quick.json (ec28f35, --context 1048576); s0: {S0} (0d91389, --context 65538)")
print("selfcons: identical", a["selfcons"]["identical"], "of", a["selfcons"]["prompts"], "| token ids of both runs equal to s0:",
      a["selfcons"]["runs"] == b["selfcons"]["runs"], "| golden hazard", a["selfcons"]["golden"]["hazard"],
      "(s0", b["selfcons"]["golden"]["hazard"], ")")
print("gsm8k:", f"{a['gsm8k']['acc']['k']}/{a['gsm8k']['acc']['n']}", "| misses equal to s0:",
      strip(a["gsm8k"]["misses"]) == strip(b["gsm8k"]["misses"]), "| mean completion tokens",
      a["gsm8k"]["mean_completion_tokens"], "(s0", b["gsm8k"]["mean_completion_tokens"], ")")
print("tools:", f"exact_args {a['tools']['exact_args']['k']}/{a['tools']['exact_args']['n']},",
      f"json_valid {a['tools']['json_valid']['k']}/{a['tools']['json_valid']['n']},",
      f"no_call {a['tools']['no_call']['k']}/{a['tools']['no_call']['n']}", "| all fields but time equal to s0:",
      strip(a["tools"]) == strip(b["tools"]))
print("needle:", f"{a['needle']['found']}/{a['needle']['total']}", "| every cell (prompt tokens, answer) equal to s0:",
      all(a["needle"]["cells"][k]["prompt_tokens"] == b["needle"]["cells"][k]["prompt_tokens"]
          and a["needle"]["cells"][k]["answer"] == b["needle"]["cells"][k]["answer"] for k in b["needle"]["cells"]))
