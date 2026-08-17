import litellm
from lib.model import MODEL

def get_model_token_limit():
    try:
        model_info = litellm.get_model_info(model=MODEL)
        token_limit = model_info.get("max_input_tokens") or 10000
        return token_limit
    except Exception:
        return 10000

