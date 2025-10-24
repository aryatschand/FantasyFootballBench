import os
import json
import dspy
from openai import OpenAI
from ffbench.scoring import calculate_fantasy_points
from ffbench.core import Player

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterLM(dspy.LM):
    def __init__(self, model_id: str):
        # Initialize with model-specific defaults to satisfy provider requirements
        default_temperature = 0.2
        default_max_tokens = 800
        mid_lower = (model_id or "").lower()
        if mid_lower.startswith("openai/gpt-5"):
            default_temperature = 1.0
            default_max_tokens = 20000
        # Pass required defaults to dspy.LM initializer so upstream checks pass
        super().__init__(model_id, temperature=default_temperature, max_tokens=default_max_tokens)
        self.model_id = model_id
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY not set")
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
        self._default_temperature = default_temperature
        self._default_max_tokens = default_max_tokens

    def __call__(self, messages=None, **kwargs):
        if messages:
            prompt = "\n".join(
                [(m["content"] if isinstance(m, dict) else getattr(m, "content", "")) for m in messages]
            ).strip()
        else:
            prompt = (kwargs.get("prompt") or "").strip()

        json_mode = kwargs.get("response_format", {}).get("type") == "json_object"
        system_content = "Respond ONLY with a valid JSON object containing the requested fields. Also include a 'rationale' field explaining your decision in natural language, for logging only."

        temperature = kwargs.get("temperature", self._default_temperature)
        max_tokens = kwargs.get("max_tokens", self._default_max_tokens)
        extra_headers = {
            "HTTP-Referer": os.environ.get("FFBENCH_SITE_URL", "https://fantasyfootballbench.local"),
            "X-Title": os.environ.get("FFBENCH_SITE_TITLE", "FantasyFootballBench"),
        }
        extra_body = {}
        mid = (self.model_id or "").lower()
        if mid.startswith("openai/gpt-5"):
            # Reasoning models often require effort and high token limits
            extra_body["reasoning"] = {"effort": "medium"}
            # Some providers honor max_output_tokens
            extra_body["max_output_tokens"] = max_tokens

        # Force GPT-5 family call-time params in addition to defaults
        if mid.startswith("openai/gpt-5"):
            temperature = 1.0
            if max_tokens < 20000:
                max_tokens = 20000

        def _do_call(max_toks):
            params = dict(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_toks,
                extra_headers=extra_headers,
                extra_body=extra_body,
            )
            # If caller requested JSON mode, ask the API to enforce it when supported
            if json_mode:
                params["response_format"] = {"type": "json_object"}
            return self.client.chat.completions.create(**params)

        try:
            completion = _do_call(max_tokens)
        except Exception:
            # One conservative retry with smaller output and without extras
            try:
                completion = _do_call(min(2048, max_tokens))
            except Exception:
                # As a last resort, return an empty JSON/text so upstream can fallback
                return {} if json_mode else ""

        text = ((completion.choices[0].message.content) or "").strip()
        if json_mode:
            try:
                return json.loads(text)
            except Exception:
                return text
        return text


class LLMManager:
    def _model_param_overrides(self, model_id: str) -> dict:
        mid = (model_id or "").lower()
        if mid.startswith("openai/gpt-5"):
            return {"temperature": 1.0, "max_tokens": 16000}
        return {}

    def call_with_signature(self, model_id, signature_class, **kwargs):
        parts = []
        for field_name in ['team_info', 'available_players', 'trade_info']:
            if field_name in kwargs:
                parts.append(f"{field_name}: {kwargs[field_name]}")
        # Output hint
        if signature_class.__name__ == "DraftPickSignature":
            parts.append('Respond ONLY with JSON: {"draft_pick": "Full Name", "rationale": "why this is optimal given roster & scoring"}')
        elif signature_class.__name__ == "ChooseStartersSignature":
            parts.append('Respond ONLY with JSON: {"starter_lineup": "Comma-separated", "rationale": "why this lineup maximizes points"}')
        elif signature_class.__name__ == "ProposeTradeSignature":
            parts.append('Respond ONLY with JSON: {"trade_proposal": "...", "rationale": "convincing reasoning for the proposal"}')
        elif signature_class.__name__ == "RespondToTradeSignature":
            parts.append('Respond ONLY with JSON: {"response": "accept|reject", "rationale": "why this decision"}')

        prompt = "\n".join(parts)
        resp = self.call_with_prompt(model_id, prompt, response_format={"type": "json_object"})
        return resp

    def call_with_prompt(self, model_id, prompt, **kwargs):
        lm = OpenRouterLM(model_id=model_id)
        # apply per-model overrides
        overrides = self._model_param_overrides(model_id)
        merged = {**kwargs, **overrides}
        return lm(prompt=prompt, **merged)

    def choose_starters(self, team, week, opponents_map, weekly_stats):
        """
        Build a start/sit prompt for the team's roster, flagging any players who
        will score exactly 0 points this week as INJURED in the prompt to discourage selection.
        Returns: list[Player] starters for this week (QB, 2 RB, 2 WR, TE, FLEX)
        """
        # Compute this week's fantasy points for each roster player
        pid_to_pts = {}
        for p in team.roster:
            rows = weekly_stats[weekly_stats.get("player_id") == p.player_id] if hasattr(weekly_stats, "get") else weekly_stats[weekly_stats["player_id"] == p.player_id]
            if not rows.empty:
                pid_to_pts[p.player_id] = float(calculate_fantasy_points(rows.iloc[0]))
            else:
                pid_to_pts[p.player_id] = 0.0

        # Build allowed names and annotated roster lines
        allowed_names = []
        annotated_lines = []
        for p in team.roster:
            pts = pid_to_pts.get(p.player_id, 0.0)
            note = " [INJURED - 0 pts this week]" if abs(pts) < 1e-9 else ""
            annotated_lines.append(f"- {p.name} ({p.position}){note}")
            allowed_names.append(p.name)

        opponent_name = list(opponents_map.values())[0] if opponents_map else "Unknown"
        team_info = (
            f"Start/Sit Decision\n"
            f"Week: {week}\n"
            f"Manager: {team.manager_name}\n"
            f"Opponent: {opponent_name}\n"
            f"Roster (players flagged as 'INJURED - 0 pts this week' should generally not be started):\n" +
            "\n".join(annotated_lines) +
            "\n\n" +
            "IMPORTANT: You are ONLY allowed to use the data explicitly provided in this prompt. "
            "Respond ONLY with a JSON object of the form {\"qb\": \"Name\", \"rbs\": [\"Name\",\"Name\"], "
            "\"wrs\": [\"Name\",\"Name\"], \"te\": \"Name\", \"flex\": \"Name\"}. "
            "All names MUST be selected from this list: " + ", ".join(allowed_names) + "."
        )

        # Ask the model for starters
        model_id = getattr(team.llm_manager, "model_id", None) or os.getenv("FFBENCH_STARTSIT_MODEL", "openai/gpt-5-mini")
        resp = self.call_with_prompt(model_id, team_info, response_format={"type": "json_object"})

        def build_best_lineup():
            # Deterministic fallback by weekly points
            players = list(team.roster)
            players_sorted = sorted(players, key=lambda x: pid_to_pts.get(x.player_id, 0.0), reverse=True)
            choose = []
            # QB
            qb = next((p for p in players_sorted if p.position == "QB"), None)
            if qb:
                choose.append(qb)
            # RB x2
            rbs = [p for p in players_sorted if p.position == "RB" and p not in choose][:2]
            choose.extend(rbs)
            # WR x2
            wrs = [p for p in players_sorted if p.position == "WR" and p not in choose][:2]
            choose.extend(wrs)
            # TE
            te = next((p for p in players_sorted if p.position == "TE" and p not in choose), None)
            if te:
                choose.append(te)
            # FLEX (best of RB/WR/TE not chosen yet)
            flex = next((p for p in players_sorted if p.position in ("RB","WR","TE") and p not in choose), None)
            if flex:
                choose.append(flex)
            return choose

        starters = []
        try:
            qb_name = (resp or {}).get("qb")
            rbs_names = (resp or {}).get("rbs", [])
            wrs_names = (resp or {}).get("wrs", [])
            te_name = (resp or {}).get("te")
            flex_name = (resp or {}).get("flex")
            pick_names = []
            if qb_name: pick_names.append(qb_name)
            pick_names.extend(rbs_names[:2])
            pick_names.extend(wrs_names[:2])
            if te_name: pick_names.append(te_name)
            if flex_name: pick_names.append(flex_name)
            # Map names back to Player objects
            name_to_player = {p.name: p for p in team.roster}
            for nm in pick_names:
                if nm in name_to_player:
                    starters.append(name_to_player[nm])
            # If incomplete, fill by best lineup
            if len(starters) < 7:
                fallback = build_best_lineup()
                for p in fallback:
                    if p not in starters and len(starters) < 7:
                        starters.append(p)
        except Exception:
            starters = build_best_lineup()
            
        return starters
    

class DraftPickSignature(dspy.Signature):
    team_info = dspy.InputField(desc="Team context")
    available_players = dspy.InputField(desc="Available players")
    draft_pick = dspy.OutputField(desc="Player full name")


class ChooseStartersSignature(dspy.Signature):
    team_info = dspy.InputField(desc="Team and matchup context")
    starter_lineup = dspy.OutputField(desc="Lineup JSON")


class ProposeTradeSignature(dspy.Signature):
    team_info = dspy.InputField(desc="Teams and stats context")
    trade_proposal = dspy.OutputField(desc="Trade JSON")


class RespondToTradeSignature(dspy.Signature):
    trade_info = dspy.InputField(desc="Proposal JSON")
    response = dspy.OutputField(desc="accept|reject")
