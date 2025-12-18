import os
import json
import re
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class DomainExpertAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Domain Expert Agent with DeepSeek Reasoner.
        Role: Senior Business Analyst / Product Owner.
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required for Domain Expert.")
        
        # Initialize OpenAI Client for DeepSeek
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com/v1"
        )
        self.model_name = "deepseek-reasoner"

    def evaluate_strategies(self, data_summary: str, business_objective: str, strategies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Critiques and scores multiple strategies based on business alignment and feasibility.
        """
        
        strategies_text = json.dumps(strategies, indent=2)
        
        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        You are a Senior Industry Expert and Business Analyst.
        Your goal is to critique technical data science proposals and select the one that delivers maximum BUSINESS VALUE.
        
        *** BUSINESS OBJECTIVE ***
        "$business_objective"
        
        *** DATA CONTEXT ***
        $data_summary
        
        *** CANDIDATE STRATEGIES ***
        $strategies_text
        
        *** YOUR TASK ***
        Evaluate each strategy (0-10 Score) based on:
        1. **Alignment:** Does it directly answer the business question? (High score) or just explore data? (Low score).
        2. **Feasibility:** Do we have the data? Is the approach realistic given the complexity?
        3. **Risk:** Is there high risk of overfitting or "black box" non-explainability where explainability is needed?
        
        *** OUTPUT FORMAT ***
        Return a JSON object with:
        {{
            "reviews": [
                {{
                    "title": "Strategy Title",
                    "score": 8.5,
                    "reasoning": "Strong alignment with pricing goal...",
                    "risks": ["Potential overfitting due to small sample"],
                    "recommendation": "Proceed with caution on feature selection."
                }},
                ...
            ]
        }}
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            business_objective=business_objective,
            data_summary=data_summary,
            strategies_text=strategies_text
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Evaluate these strategies."}
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
            return json.loads(cleaned_content)
            
        except Exception as e:
            print(f"Domain Expert Error: {e}")
            # Fallback: Return empty reviews, graph will handle selection fallback
            return {"reviews": []}

    def _clean_json(self, text: str) -> str:
        text = re.sub(r'```json', '', text)
        text = re.sub(r'```', '', text)
        return text.strip()
