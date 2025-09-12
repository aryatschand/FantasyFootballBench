#!/usr/bin/env python3
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ffbench.llm import LLMManager


def test_bedrock_models_connectivity():
    models_csv = os.environ.get("BEDROCK_MODELS_CSV", "")
    default_model = "arn:aws:bedrock:us-east-2:851725383897:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"
    models = [m.strip() for m in models_csv.split(",") if m.strip()] if models_csv.strip() else [default_model for _ in range(10)]

    llm = LLMManager()
    for idx, model_id in enumerate(models):
        resp = llm.call_with_prompt(model_id, "Return the word READY only.")
        assert isinstance(resp, str) and "ready" in resp.lower(), f"Model {idx+1} did not respond READY. Got: {resp}"


