# Configuration for the Large Language Model (LLM) API

# Replace this URL with the actual endpoint of your locally running LLM
# (e.g., from text-generation-webui, Ollama, or other OpenAI-compatible server)
# Example for text-generation-webui default: "http://localhost:5000/v1"
# Example for Ollama default (if using openai compatibility): "http://localhost:11434/v1"
LLM_API_ENDPOINT = "http://localhost:5000/v1"

# You might also want to specify a model name if your server requires it
# or if you want to ensure a specific model is used.
# This depends on the LLM server's API. For OpenAI-compatible APIs,
# the model is often specified in the request body.
# LLM_MODEL_NAME = "local-model"
