#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.llm import LLMManager
from ffbench.config import get_models


def main():
    llm = LLMManager()
    models = get_models()
    ok = []
    bad = []
    for m in models:
        arn = m["arn"]
        try:
            resp = llm.call_with_prompt(arn, "reply 'ok'")
            if isinstance(resp, str) and "ok" in resp.lower():
                ok.append(m)
            else:
                ok.append(m)  # treat any response as success
        except Exception as e:
            bad.append((m, str(e)))
    print(f"OK: {len(ok)}; BAD: {len(bad)}")
    for m, err in bad:
        print(f"FAILED: {m['name']} {m['arn']} -> {err}")


if __name__ == "__main__":
    main()


