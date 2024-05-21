from src.config import *
from typing import Any

def json2token(obj : Any, update_special_tokens_for_json_key: bool = False, sort_json_key: bool = True) -> Any:
        """
        Convert an ordered JSON object into a token sequence
        """
        if type(obj) == dict:
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            else:
                output = ""
                if sort_json_key:
                    keys = sorted(obj.keys(), reverse=True)
                else:
                    keys = obj.keys()
                for k in keys:
                    if update_special_tokens_for_json_key:
                        new_special_tokens.append(fr"<s_{k}>") if fr"<s_{k}>" not in new_special_tokens else None
                        new_special_tokens.append(fr"</s_{k}>") if fr"</s_{k}>" not in new_special_tokens else None
                    output += (
                        fr"<s_{k}>"
                        + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key)
                        + fr"</s_{k}>"
                    )
                return output
        elif type(obj) == list:
            return r"<sep/>".join(
                [json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj]
            )
        else:
            # excluded special tokens for now
            obj = str(obj)
            if f"<{obj}/>" in params['special_tokens']:
                obj = f"<{obj}/>"  # for categorical special tokens
            return obj