import requests
import json
import os
from webapp.llm_config import LLM_API_ENDPOINT

BLACKLIST_FILE = os.path.join(os.path.dirname(__file__), "blacklist.txt")
BLACKLISTED_WORDS = []

def load_blacklist():
    global BLACKLISTED_WORDS
    if os.path.exists(BLACKLIST_FILE):
        with open(BLACKLIST_FILE, 'r') as f:
            BLACKLISTED_WORDS = [line.strip().lower() for line in f if line.strip()]
        if BLACKLISTED_WORDS:
            print(f"Loaded {len(BLACKLISTED_WORDS)} words from blacklist: {BLACKLIST_FILE}")
        else:
            print(f"Blacklist file found but is empty or contains no valid words: {BLACKLIST_FILE}")
    else:
        print(f"Blacklist file not found: {BLACKLIST_FILE}. No content filtering will be applied.")

load_blacklist() # Load blacklist when module is imported

# Placeholder for the model name that the local LLM server expects.
# Users might need to change this depending on how they've loaded the model
# in their serving tool (e.g., text-generation-webui, Ollama).
DEFAULT_MODEL_NAME = "local-model"

def _filter_content(text_response: str) -> str:
    if BLACKLISTED_WORDS:
        for word in BLACKLISTED_WORDS:
            if word in text_response.lower():
                print(f"Blacklisted word '{word}' found in LLM response. Applying filter.")
                return "[Content filtered due to potentially sensitive subject matter.]"
    return text_response

def query_llm(prompt: str, model_name: str = DEFAULT_MODEL_NAME) -> str:
    """
    Queries the locally hosted LLM with the given prompt.

    Args:
        prompt: The text prompt to send to the LLM.
        model_name: The name of the model to use (specific to the LLM server setup).

    Returns:
        The LLM's response text, or an error message if something goes wrong.
    """
    headers = {
        "Content-Type": "application/json",
    }

    # OpenAI-compatible chat completions payload
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7, # Adjust temperature as needed
        # "max_tokens": 150,  # Optional: limit response length
    }

    try:
        response = requests.post(LLM_API_ENDPOINT, headers=headers, data=json.dumps(payload), timeout=60) # 60 seconds timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_json = response.json()

        # Try to extract content from choices, typical for OpenAI-like APIs
        if response_json.get("choices") and len(response_json["choices"]) > 0:
            message = response_json["choices"][0].get("message")
            if message and message.get("content"):
                return _filter_content(message["content"].strip())
            # Fallback for some other structures if message.content is not found
            if response_json["choices"][0].get("text"):
                 return _filter_content(response_json["choices"][0]["text"].strip())

        # Fallback for APIs that might return text directly (less common for OpenAI standard)
        if response_json.get("text"):
            return _filter_content(response_json.get("text").strip())

        return "Error: Could not parse LLM response structure. Response: " + json.dumps(response_json)

    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {e}"
    except json.JSONDecodeError:
        return f"Error: Could not decode JSON response from LLM. Response content: {response.text}"
    except KeyError:
        return f"Error: Unexpected response structure from LLM. Response: " + json.dumps(response_json)
    except Exception as e:
        return f"Error: An unexpected error occurred: {e}"

if __name__ == '__main__':
    # Simple test (requires your LLM server to be running and configured in llm_config.py)
    test_prompt = "Explain the concept of a Large Language Model in one sentence."
    print(f"Sending prompt: '{test_prompt}' to {LLM_API_ENDPOINT} using model '{DEFAULT_MODEL_NAME}'")
    response = query_llm(test_prompt)
    print(f"LLM Response: {response}")

    test_prompt_2 = "What is the capital of France?"
    print(f"Sending prompt: '{test_prompt_2}' to {LLM_API_ENDPOINT} using model '{DEFAULT_MODEL_NAME}'")
    response_2 = query_llm(test_prompt_2)
    print(f"LLM Response: {response_2}")
