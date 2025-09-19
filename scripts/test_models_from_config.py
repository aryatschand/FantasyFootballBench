#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.llm import LLMManager
from ffbench.config import get_models


def main():
    if not os.getenv("OPENROUTER_API_KEY"):
        print("Missing OPENROUTER_API_KEY")
        return
    llm = LLMManager()
    models = get_models()
    ok = []
    bad = []
    for m in models:
        mid = m["id"]
        try:
            resp = llm.call_with_prompt(mid, 'Respond ONLY with JSON: {"ok": true}', response_format={"type":"json_object"})
            if (isinstance(resp, dict) and resp.get("ok") is True) or (isinstance(resp, str) and "ok" in resp.lower()):
                ok.append(m)
            else:
                ok.append(m)
        except Exception as e:
            bad.append((m, str(e)))
    print(f"OK: {len(ok)}; BAD: {len(bad)}")
    for m, err in bad:
        print(f"FAILED: {m['name']} {m.get('id','')} -> {err}")


if __name__ == "__main__":
    main()


