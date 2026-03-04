"""
Simplified functions to get the files and architecture of all models in the Hugging Face cache.
"""

from huggingface_hub import scan_cache_dir
import json

def get_all_models_files() -> dict[str, list[str]]:
    """
    Returns a dictionary where the key is the model name (repo_id)
    and the value is the list of paths to its files in the cache.
    """
    cache = scan_cache_dir()
    models_files = {}
    for repo in cache.repos:
        repo_id = repo.repo_id
        if 'main' in repo.refs:
            models_files[repo_id] = [file.file_path for file in repo.refs['main'].files]
        else:
            models_files[repo_id] = [file.file_path for file in repo.files]
    return models_files

def get_all_models_architectures(arch_filter: str = None) -> dict[str, list[str]]:
    """
    Returns a dictionary where the key is the model name (repo_id)
    and the value is its architecture read from its `config.json` file.
    If `arch_filter` is provided, only models containing that substring in any of their architectures are returned.
    """
    cache = scan_cache_dir()
    models_architectures = {}
    for repo in cache.repos:
        repo_id = repo.repo_id
        files = repo.refs['main'].files if 'main' in repo.refs else repo.files
        architecture = []
        for file in files:
            if file.file_name == "config.json":
                try:
                    with open(file.file_path, 'r', encoding='utf-8') as f:
                        architecture = json.load(f).get('architectures', [])
                except (json.JSONDecodeError, OSError):
                    pass
                break # We already found the config.json for this repo
        
        if arch_filter is None or any(arch_filter in a for a in architecture):
            models_architectures[repo_id] = architecture
    return models_architectures

if __name__ == "__main__":
    print("--- Files per model ---")
    all_files = get_all_models_files()
    for model, files in all_files.items():
        print(f"Model: {model}")
        print(f"Number of files: {len(files)}")
        # To avoid printing massive lists, we only print the first 3 files
        for f in files[:3]:
            print(f"  - {f}")
        if len(files) > 3:
            print(f"  ... and {len(files) - 3} more files.")
        print()

    print("--- Architectures per model ---")
    all_archs = get_all_models_architectures()
    for model, arch in all_archs.items():
        print(f"{model}: {arch}")

    # Print models based in Qwen2_5 architecture
    print("\n--- Models based in Qwen2_5 architecture ---")
    qwen_archs = get_all_models_architectures(arch_filter="Qwen2_5_VL")
    for model in qwen_archs:
        print(model)    

    # Print models based in Qwen3VL architecture
    print("\n--- Models based in Qwen3VL architecture ---")
    qwen_archs = get_all_models_architectures(arch_filter="Qwen3VL")
    for model in qwen_archs:
        print(model)    
