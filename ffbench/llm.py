import pandas as pd
from openai import OpenAI
import os

class LLMManager:
    def __init__(self, model_name="gpt-4-turbo"):
        self.model_name = model_name
        # It's recommended to set the API key as an environment variable
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _get_llm_response(self, prompt):
        """
        Sends a prompt to the LLM and gets a response.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7, # Add some creativity
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error communicating with OpenAI: {e}")
            return None # Fallback or error handling

    def make_draft_pick(self, team, available_players, historical_data):
        """
        Generates a prompt for the LLM to make a draft pick.
        """
        prompt = f"""
        You are the manager of team "{team.manager_name}".
        It is week {historical_data.week}. Your matchup is against {historical_data.opponent_name}.
        Your roster is:
        {pd.DataFrame([p.__dict__ for p in team.roster]).to_string()}

        Based on this information, who would you like to draft? Please respond with the player's full name.
        """
        
        llm_response = self._get_llm_response(prompt)
        
        # Simple parsing: find the player name in the response
        if llm_response:
            for player_name in available_players['player_name']:
                if player_name in llm_response:
                    return available_players[available_players['player_name'] == player_name].iloc[0]

        return available_players.iloc[0] # Fallback to top available player

    def choose_starters(self, team, week, weekly_matchups):
        """
        Generates a prompt for the LLM to choose starters.
        """
        prompt = f"""
        You are the manager of team "{team.manager_name}".
        It is week {week}. Your matchup is against {weekly_matchups[team.team_id]}.
        Your roster is:
        {pd.DataFrame([p.__dict__ for p in team.roster]).to_string()}

        Select your starting lineup (1 QB, 2 RB, 2 WR, 1 TE, 1 FLEX).
        Respond with a comma-separated list of player names.
        """
        
        llm_response = self._get_llm_response(prompt)
        
        # Parse response to get starters
        if llm_response:
            starter_names = [name.strip() for name in llm_response.split(',')]
            starters = [p for p in team.roster if p.name in starter_names]
            if len(starters) > 5: # Basic validation
                return starters

        # Fallback to heuristic
        starters = []
        qb_candidates = [p for p in team.roster if p.position == 'QB']
        if qb_candidates:
            starters.append(qb_candidates[0])
        rb_candidates = [p for p in team.roster if p.position == 'RB']
        if rb_candidates:
            starters.append(rb_candidates[0])
            starters.append(rb_candidates[1])
        wr_candidates = [p for p in team.roster if p.position == 'WR']
        if wr_candidates:
            starters.append(wr_candidates[0])
            starters.append(wr_candidates[1])
        te_candidates = [p for p in team.roster if p.position == 'TE']
        if te_candidates:
            starters.append(te_candidates[0])
        flex_candidates = [p for p in team.roster if p.position == 'FLEX']
        if flex_candidates:
            starters.append(flex_candidates[0])
            
        return starters
    
    def propose_trade(self, team, opponent_team):
        """
        Generates a prompt for the LLM to propose a trade.
        """
        prompt = f"""
        You are the manager of team "{team.manager_name}".
        You are proposing a trade with team "{opponent_team.manager_name}".
        Your roster is:
        {pd.DataFrame([p.__dict__ for p in team.roster]).to_string()}
        Their roster is:
        {pd.DataFrame([p.__dict__ for p in opponent_team.roster]).to_string()}

        Players to offer: [player1, player2, ...]
        Players to request: [playerA, playerB, ...]
        If you don't want to propose a trade, respond with "No trade".
        """
        llm_response = self._get_llm_response(prompt)

        if llm_response and "No trade" not in llm_response:
            # Basic parsing of the trade proposal
            try:
                lines = llm_response.split('\n')
                offered_line = [line for line in lines if "Players to offer:" in line][0]
                requested_line = [line for line in lines if "Players to request:" in line][0]

                offered_names = [name.strip() for name in offered_line.split(':')[1].replace('[', '').replace(']', '').split(',')]
                requested_names = [name.strip() for name in requested_line.split(':')[1].replace('[', '').replace(']', '').split(',')]

                players_offered = [p for p in team.roster if p.name in offered_names]
                players_requested = [p for p in opponent_team.roster if p.name in requested_names]

                if players_offered and players_requested:
                    return {"offered": players_offered, "requested": players_requested}
            except Exception as e:
                print(f"Error parsing trade response: {e}")

        return None

    def respond_to_trade(self, team, trade_proposal):
        """
        Generates a prompt for the LLM to respond to a trade proposal.
        """
        prompt = f"""
        You are the manager of team "{team.manager_name}".
        You have received a trade proposal from team "{trade_proposal['opponent_name']}".
        Their offer: {pd.DataFrame([p.__dict__ for p in trade_proposal['offered']]).to_string()}
        Their request: {pd.DataFrame([p.__dict__ for p in trade_proposal['requested']]).to_string()}

        Do you accept or reject this trade? Respond with "accept" or "reject".
        """
        llm_response = self._get_llm_response(prompt)
        
        if llm_response and "accept" in llm_response.lower():
            return "accept"
        
        return "reject" # Default to rejecting 