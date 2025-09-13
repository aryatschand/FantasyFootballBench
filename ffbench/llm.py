import pandas as pd
import boto3
import json
from botocore.exceptions import ClientError
import dspy

class BedrockLM(dspy.LM):
    def __init__(self, model_id="arn:aws:bedrock:us-east-2:851725383897:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0", region_name="us-east-2"):
        super().__init__(model_id)
        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region_name)

    def __call__(self, messages=None, **kwargs):
        if messages:
            # Extract the prompt from messages
            prompt = ""
            for msg in messages:
                if isinstance(msg, dict) and "content" in msg:
                    prompt += msg["content"] + "\n"
                elif hasattr(msg, 'content'):
                    prompt += msg.content + "\n"
        else:
            # Fallback for direct prompt
            prompt = kwargs.get("prompt", "")

        # Check if we need JSON mode (for DSPy structured output)
        is_json_mode = kwargs.get("response_format", {}).get("type") == "json_object"

        # For DSPy signatures, we want structured responses
        if hasattr(self, '_dspy_mode') and self._dspy_mode:
            is_json_mode = True

        # Detect provider from model_id
        mid = (self.model_id or "").lower()
        is_anthropic = "anthropic" in mid
        is_llama = ("llama" in mid) or ("meta." in mid) or ("us.meta." in mid)

        # Build payload based on provider
        if is_anthropic:
            if is_json_mode:
                content = f"{prompt.strip()}\n\nRespond with a valid JSON object containing the expected fields."
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get("max_tokens", 2000),
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": content}],
                    "response_format": {"type": "json_object"}
                }
            else:
                payload = {
                    "anthropic_version": "bedrock-2023-05-31",
                    "max_tokens": kwargs.get("max_tokens", 1000),
                    "thinking": {"type": "disabled"},
                    "messages": [{"role": "user", "content": prompt.strip()}]
                }
        elif is_llama:
            # Meta Llama instruct expects 'prompt'
            llama_prompt = prompt.strip()
            if is_json_mode:
                llama_prompt += "\n\nRespond ONLY with a valid JSON object containing the expected fields."
            payload = {
                "prompt": llama_prompt,
                "max_gen_len": kwargs.get("max_gen_len", 512),
                "temperature": kwargs.get("temperature", 0.5),
            }
        else:
            # Default fallback: simple prompt schema
            payload = {"prompt": prompt.strip()}

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload)
        )
        result = json.loads(response["body"].read())

        # Parse response
        if is_anthropic:
            if is_json_mode:
                try:
                    text_response = result["content"][0]["text"] if result.get("content") else ""
                    return json.loads(text_response)
                except (json.JSONDecodeError, IndexError, KeyError):
                    return result["content"][0]["text"] if result.get("content") else ""
            else:
                return result["content"][0]["text"] if result.get("content") else ""
        elif is_llama:
            text = result.get("generation", "")
            if is_json_mode:
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return text
            return text
        else:
            # Unknown provider, return raw
            return json.dumps(result)

class LLMManager:
    def __init__(self):
        """
        Initialize LLMManager without a specific model.
        Models are specified per call for flexibility.
        """
        pass

    def call_with_signature(self, model_id, signature_class, **kwargs):
        """
        Generic method to call LLM with any DSPy signature.

        Args:
            model_id (str): AWS Bedrock model ID to use
            signature_class: DSPy signature class
            **kwargs: Arguments to pass to the signature

        Returns:
            Parsed result object with the expected fields
        """
        # Build the prompt from the signature and inputs
        prompt_parts = []

        # Add input fields to prompt
        for field_name in ['team_info', 'available_players', 'trade_info']:
            if field_name in kwargs:
                field_desc = getattr(signature_class, field_name, None)
                if field_desc and hasattr(field_desc, 'desc'):
                    prompt_parts.append(f"{field_desc.desc}: {kwargs[field_name]}")
                else:
                    prompt_parts.append(f"{field_name.replace('_', ' ').title()}: {kwargs[field_name]}")

        # Add output instructions based on signature type
        if signature_class == DraftPickSignature:
            prompt_parts.append("\nRespond with the draft_pick (just the player's full name).")
        elif signature_class == ChooseStartersSignature:
            prompt_parts.append("\nRespond with the starter_lineup (comma-separated player names).")
        elif signature_class == ProposeTradeSignature:
            prompt_parts.append("\nRespond with the trade_proposal (players to offer and request format).")
        elif signature_class == RespondToTradeSignature:
            prompt_parts.append("\nRespond with the response (either 'accept' or 'reject').")

        prompt = "\n".join(prompt_parts)

        # Call with direct prompt instead of DSPy
        response = self.call_with_prompt(model_id, prompt)

        # Parse the response based on signature type
        if signature_class == DraftPickSignature:
            return type('Result', (), {'draft_pick': response.strip()})()
        elif signature_class == ChooseStartersSignature:
            return type('Result', (), {'starter_lineup': response.strip()})()
        elif signature_class == ProposeTradeSignature:
            return type('Result', (), {'trade_proposal': response.strip()})()
        elif signature_class == RespondToTradeSignature:
            return type('Result', (), {'response': response.strip().lower()})()
        else:
            # Generic response
            return type('Result', (), {'response': response.strip()})()

    def call_with_prompt(self, model_id, prompt, **kwargs):
        """
        Direct prompt-based call without DSPy signature.

        Args:
            model_id (str): AWS Bedrock model ID to use
            prompt (str): The prompt to send to the model
            **kwargs: Additional parameters for the model

        Returns:
            str: Model response
        """
        lm = BedrockLM(model_id=model_id)
        return lm(prompt=prompt, **kwargs)


# Example DSPy signatures for common fantasy football tasks
# Note: These are provided for reference but the LLMManager uses a simplified approach
class DraftPickSignature(dspy.Signature):
    """Make a draft pick based on team info and available players."""
    team_info = dspy.InputField(desc="Information about the team and current situation")
    available_players = dspy.InputField(desc="List of available players to choose from")
    draft_pick = dspy.OutputField(desc="The full name of the player to draft")


class ChooseStartersSignature(dspy.Signature):
    """Choose starting lineup based on team roster and matchup."""
    team_info = dspy.InputField(desc="Information about the team and current matchup")
    starter_lineup = dspy.OutputField(desc="Comma-separated list of player names for starting lineup (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX)")


class ProposeTradeSignature(dspy.Signature):
    """Propose a trade based on team rosters."""
    team_info = dspy.InputField(desc="Information about your team and opponent's roster")
    trade_proposal = dspy.OutputField(desc="Trade proposal with players to offer and request, or 'No trade' if none")


class RespondToTradeSignature(dspy.Signature):
    """Respond to a trade proposal."""
    trade_info = dspy.InputField(desc="Information about the trade proposal")
    response = dspy.OutputField(desc="Either 'accept' or 'reject'")


# Usage Examples:
"""
# Initialize the LLM manager
llm_manager = LLMManager()

# Define your model ID
model_id = "arn:aws:bedrock:us-east-2:851725383897:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0"

# Example 1: Draft pick
result = llm_manager.call_with_signature(
    model_id=model_id,
    signature_class=DraftPickSignature,
    team_info="You need a QB for your fantasy team",
    available_players="Josh Allen, Patrick Mahomes, Joe Burrow"
)
print(f"Draft pick: {result.draft_pick}")

# Example 2: Direct prompt
response = llm_manager.call_with_prompt(
    model_id=model_id,
    prompt="What are the best strategies for fantasy football?"
)
print(f"Strategy advice: {response}")

# Example 3: Custom signature
class CustomSignature(dspy.Signature):
    question = dspy.InputField(desc="The question to answer")
    answer = dspy.OutputField(desc="The answer to the question")

result = llm_manager.call_with_signature(
    model_id=model_id,
    signature_class=CustomSignature,
    question="Who will win the Super Bowl?"
)
print(f"Prediction: {result.answer}")
""" 