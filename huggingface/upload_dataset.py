import os
from huggingface_hub import HfApi

# Configuration
print("Please enter your Hugging Face token:")
token = input()
api = HfApi(token=token)

# Upload du dossier
api.upload_folder(
    folder_path=".",
    repo_id="kabsis/NeurofluxModels",
    repo_type="dataset",
    path_in_repo="",
    commit_message="Initial commit: NeuroFlux Models repository"
)

print("Successfully uploaded to Hugging Face!")
