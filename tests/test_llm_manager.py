#!/usr/bin/env python3
"""
Test the new simplified LLMManager with DSPy signatures.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ffbench.llm import (
    LLMManager,
    DraftPickSignature,
    ChooseStartersSignature,
    ProposeTradeSignature,
    RespondToTradeSignature
)

def test_llm_manager():
    """Test the new LLMManager functionality."""
    print("🧪 Testing simplified LLMManager...")

    # Initialize the manager
    llm_manager = LLMManager()

    # Test model ID
    model_id = "arn:aws:bedrock:us-east-2:851725383897:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

    # Test 1: Direct prompt call
    print("\n1. Testing direct prompt call...")
    prompt = "What is 2 + 2? Respond with just the number."
    try:
        response = llm_manager.call_with_prompt(model_id, prompt)
        print(f"✅ Direct prompt response: {response}")
    except Exception as e:
        print(f"❌ Direct prompt failed: {e}")

    # Test 2: DSPy signature call
    print("\n2. Testing DSPy signature call...")
    try:
        result = llm_manager.call_with_signature(
            model_id=model_id,
            signature_class=DraftPickSignature,
            team_info="You are managing a fantasy football team. You need a QB.",
            available_players="Josh Allen (QB), Patrick Mahomes (QB), Joe Burrow (QB)"
        )
        if result:
            print(f"✅ Signature response - Draft pick: {result.draft_pick}")
        else:
            print("❌ Signature call returned None")
    except Exception as e:
        print(f"❌ Signature call failed: {e}")

    # Test 3: Different signature
    print("\n3. Testing starter selection signature...")
    try:
        result = llm_manager.call_with_signature(
            model_id=model_id,
            signature_class=ChooseStartersSignature,
            team_info="Your team has: Josh Allen (QB), Christian McCaffrey (RB), Davante Adams (WR), Travis Kelce (TE)"
        )
        if result:
            print(f"✅ Starter lineup: {result.starter_lineup}")
        else:
            print("❌ Starter selection returned None")
    except Exception as e:
        print(f"❌ Starter selection failed: {e}")

    print("\n🎉 LLMManager tests completed!")

if __name__ == "__main__":
    test_llm_manager()
