from string import Template
from typing import Any, Dict

def render_prompt(template_str: str, **kwargs: Any) -> str:
    """
    Safely renders a prompt using string.Template to prevent f-string injection attacks
    and accidental crashes due to braces in the content (e.g. JSON or regex).
    
    Args:
        template_str (str): The template string with $variable placeholders.
        **kwargs: Variables to substitute into the template.
        
    Returns:
        str: The rendered prompt.
    """
    tpl = Template(template_str)
    # Convert all kwargs to string to ensure safe substitution
    str_kwargs = {k: str(v) for k, v in kwargs.items()}
    return tpl.safe_substitute(**str_kwargs)
