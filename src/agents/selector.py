import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class SelectorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Selector Agent with MIMO v2 Flash.
        """
        self.api_key = api_key or os.getenv("MIMO_API_KEY")
        if not self.api_key:
            raise ValueError("MIMO API Key is required.")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.xiaomimimo.com/v1",
        )
        self.model_name = "mimo-v2-flash"

    def select_best_strategy(self, strategies: Dict[str, Any], data_audit: str) -> Dict[str, Any]:
        """
        Selects the best strategy from the list based on data viability and impact.
        Returns a dictionary with 'best_strategy_index' and 'reason'.
        """
        strategies_list = strategies.get("strategies", [])
        if not strategies_list:
            return {"best_strategy_index": 0, "reason": "No strategies provided."}

        strategies_text = json.dumps(strategies_list, indent=2)

        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        Eres un CTO Senior evaluando propuestas de Data Science. 
        Tienes el reporte de datos (Audit) y una lista de estrategias candidatas. 
        
        TU OBJETIVO: Seleccionar la estrategia mケs viable y con mayor impacto. 
        
        CRITERIOS:
        1. Viabilidad de Datos: 隅Las columnas requeridas existen en el Audit? (Si faltan columnas crヴticas, DESCARTA la estrategia).
        2. Claridad: 隅La hipИtesis es sИlida?
        3. Riesgo: Prefiere 'Low' o 'Medium' difficulty si el impacto es similar.

        DATA AUDIT (GROUND TRUTH):
        $data_audit

        CANDIDATE STRATEGIES:
        $strategies_text

        SALIDA: Devuelve れNICAMENTE un JSON con este formato: 
        {{
            "best_strategy_index": <int>, 
            "reason": "<string>"
        }}
        """

        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            data_audit=data_audit,
            strategies_text=strategies_text,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Select the best strategy and return JSON only."},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            cleaned_content = self._clean_json(content)
            return json.loads(cleaned_content)
        except Exception as e:
            print(f"Selector Error: {e}. Defaulting to index 0.")
            return {
                "best_strategy_index": 0,
                "reason": f"Selection failed: {e}",
            }

    def _clean_json(self, text: str) -> str:
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)
        return text.strip()
