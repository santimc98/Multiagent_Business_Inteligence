import google.generativeai as genai
import os
import json
import re
from typing import List, Dict, Any
from dotenv import load_dotenv
from google.generativeai.types import HarmCategory, HarmBlockThreshold

load_dotenv()

class SelectorAgent:
    def __init__(self, api_key: str = None):
        """
        Initializes the Selector Agent with Gemini 2.5 Flash.
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Google API Key is required.")
        
        genai.configure(api_key=self.api_key)
        
        # Configuration
        generation_config = {
            "temperature": 0.1, # Low temperature for logical decision making
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "application/json",
        }

        # Safety Settings
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-flash-lite", # Migrate to 2.5 Flash (Available Quota)
            generation_config=generation_config,
            safety_settings=self.safety_settings
        )

    def select_best_strategy(self, strategies: Dict[str, Any], data_audit: str) -> Dict[str, Any]:
        """
        Selects the best strategy from the list based on data viability and impact.
        Returns a dictionary with 'best_strategy_index' and 'reason'.
        """
        
        strategies_list = strategies.get('strategies', [])
        if not strategies_list:
             return {"best_strategy_index": 0, "reason": "No strategies provided."}

        strategies_text = json.dumps(strategies_list, indent=2)

        from src.utils.prompting import render_prompt

        SYSTEM_PROMPT_TEMPLATE = """
        Eres un CTO Senior evaluando propuestas de Data Science. 
        Tienes el reporte de datos (Audit) y una lista de estrategias candidatas. 
        
        TU OBJETIVO: Seleccionar la estrategia más viable y con mayor impacto. 
        
        CRITERIOS:
        1. Viabilidad de Datos: ¿Las columnas requeridas existen en el Audit? (Si faltan columnas críticas, DESCARTA la estrategia).
        2. Claridad: ¿La hipótesis es sólida?
        3. Riesgo: Prefiere 'Low' o 'Medium' difficulty si el impacto es similar.

        DATA AUDIT (GROUND TRUTH):
        $data_audit

        CANDIDATE STRATEGIES:
        $strategies_text

        SALIDA: Devuelve ÚNICAMENTE un JSON con este formato: 
        {{
            "best_strategy_index": <int>, 
            "reason": "<string>"
        }}
        """
        
        system_prompt = render_prompt(
            SYSTEM_PROMPT_TEMPLATE,
            data_audit=data_audit,
            strategies_text=strategies_text
        )
        
        try:
            response = self.model.generate_content(system_prompt)
            result = json.loads(response.text)
            return result
        except Exception as e:
            print(f"Selector Error: {e}. Defaulting to index 0.")
            return {
                "best_strategy_index": 0,
                "reason": f"Selection failed: {e}"
            }
