import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class StrategistAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Strategist Agent with DeepSeek Reasoner.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required.")
        
        # Initialize OpenAI Client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name = "deepseek-reasoner"

    def generate_strategies(self, data_summary: str, user_request: str) -> Dict[str, Any]:
        """
        Generates analysis strategies based on the data summary and user request.
        """
        
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Chief Data Strategist. Your goal is to formulate a precise analysis strategy given a dataset summary and a user business question.
        
        *** DATASET SUMMARY ***
        $data_summary
        
        *** USER REQUEST ***
        "$user_request"
        
        *** YOUR TASK ***
        1. Analyze the user's intent (e.g., "Why are we losing money?", "Predict churn", "Cluster customers").
        2. BRAINSTORM 3 DISTINCT STRATEGIES to solve this problem:
           - **Strategy 1 (Quick Win):** Simple, fast, high interpretability (e.g. Correlation/Regression).
           - **Strategy 2 (Advanced):** High performance, predictive power (e.g. Random Forest/XGBoost/Clustering).
           - **Strategy 3 (Creative/Alternative):** A different angle (e.g. Segmentation if asked for Churn, or Time Series if applicable).
        
        *** DATA SCIENCE FIRST PRINCIPLES (UNIVERSAL REASONING) ***
        1. **REPRESENTATIVENESS (The "Bias" Check):**
           - Does your selected data subset represent the *Full Reality* of the problem?
           - *Rule:* NEVER filter the target variable to a single class if the goal is comparison or prediction.
        
        2. **SIGNAL MAXIMIZATION (The "Feature" Check):**
           - *Action:* Select ALL columns that might carry information. Be broad.
           
        3. **TARGET CLARITY:**
           - What exactly are we solving for? (e.g. Price Optimization -> Target = "Success Probability" given Price).
           
        *** CRITICAL OUTPUT RULES ***
        - RETURN ONLY RAW JSON. NO MARKDOWN. NO COMMENTS.
        - The output must be a dictionary with a single key "strategies" containing a LIST of 3 objects.
        - Each object keys: "title", "analysis_type", "hypothesis", "required_columns" (list of strings), "estimated_difficulty", "reasoning".
        - "required_columns": Use EXACT column names from the summary.
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            data_summary=data_summary,
            user_request=user_request
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Generate the strategy JSON."}
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={'type': 'json_object'},
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            cleaned_content = self._clean_json(content)
            single_strategy = json.loads(cleaned_content)
            
            return single_strategy
            
        except Exception as e:
            print(f"Strategist Error: {e}")
            # Fallback simple strategy
            return {"strategies": [{
                "title": "Error Fallback Strategy",
                "analysis_type": "statistical",
                "hypothesis": "Could not generate complex strategy. Analyzing basic correlations.",
                "required_columns": [], 
                "estimated_difficulty": "Low", 
                "reasoning": f"DeepSeek API Failed: {e}"
            }]}

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()
