import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    print("Error: No se encontró GOOGLE_API_KEY en el archivo .env")
else:
    genai.configure(api_key=api_key)
    print("Conectando con Google AI Studio...")
    try:
        print("\n--- Modelos Disponibles ---")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Nombre: {m.name} | Display: {m.display_name}")
    except Exception as e:
        print(f"Error al listar modelos: {e}")