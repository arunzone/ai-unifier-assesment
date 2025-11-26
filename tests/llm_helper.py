import json


def ai_response_for(arguments: str):
    return {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": f"{arguments}",
                    "refusal": None,
                    "annotations": [],
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
    }


def extract_user_content(request_content: str) -> str:
    try:
        data = json.loads(request_content)
        user_content = data["messages"][-1]["content"]
        return user_content
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Error parsing request: {e}")
        return ""
