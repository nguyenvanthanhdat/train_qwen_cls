#sigmaloop/checkpoint-116000

from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sigmaloop/checkpoint-116000", 
    local_dir=".", 
    local_dir_use_symlinks=False,
    cache_dir=".cache"
)

#snapshot_download(repo_id="google/fleurs", repo_type="dataset")