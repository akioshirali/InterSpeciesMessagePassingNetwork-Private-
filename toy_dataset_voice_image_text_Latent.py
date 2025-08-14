import os
import json
import numpy as np
from typing import List, Tuple, Union
from datetime import datetime

# --------------------------
# Encoder registry (extend freely)
# --------------------------
ENCODERS = {
    # Image encoders
    "Image:ResNet50":       {"encoder": "ResNet50 (ImageNet)",                "latent_dim": 2048},
    "Image:CLIP-ViT-B32":   {"encoder": "CLIP ViT-B/32",                      "latent_dim": 512},
    "Image:EfficientNetB4": {"encoder": "EfficientNet-B4",                    "latent_dim": 1792},

    # Text encoders
    "Text:BERT":            {"encoder": "BERT-base-uncased",                  "latent_dim": 768},
    "Text:SentenceVec":     {"encoder": "SentenceTransformer all-MiniLM-L6-v2","latent_dim": 384},

    # Video encoder
    "Video:TimeSformer":    {"encoder": "TimeSformer",                         "latent_dim": 1024},

    # Voice encoders
    "Voice:Wav2Vec2":       {"encoder": "Wav2Vec2-base",                       "latent_dim": 768},
    "Voice:Whisper":        {"encoder": "OpenAI Whisper-small",                "latent_dim": 1024},
    "Voice:ECAPA-TDNN":     {"encoder": "ECAPA-TDNN",                          "latent_dim": 192}
}

# --------------------------
# Helpers
# --------------------------
def _sanitize(s: str) -> str:
    """Filesystem-safe token (keeps it readable)."""
    return (
        s.replace(" ", "")
         .replace("/", "-")
         .replace("\\", "-")
         .replace(":", "-")
    )

def _combo_label(info_types: List[str]) -> str:
    # Example: Image-CLIP-ViT-B32__Text-SentenceVec
    parts = []
    for t in info_types:
        modality, name = t.split(":", 1)
        parts.append(f"{modality}-{name}")
    return "__".join(_sanitize(p) for p in parts)

def _space_label(same_latent_space: bool, dims: List[int]) -> str:
    if same_latent_space:
        return f"same_latent__{max(dims)}d"
    else:
        return "separate_latent"

def _file_label(tag: str, n: int, dim: int) -> str:
    # Example: Image__CLIP-ViT-B32__n=10000__dim=512.npy
    modality, name = tag.split(":", 1)
    return f"{_sanitize(modality)}__{_sanitize(name)}__n={n}__dim={dim}.npy"

# --------------------------
# Core generator + saver
# --------------------------
def generate_toy_dataset_to_disk(
    info_types: List[str],
    num_examples: Union[int, List[int]],
    same_latent_space: bool,
    out_dir: str,
    write_readme: bool = True
) -> Tuple[List[str], dict]:
    """
    Generate toy embeddings and save them as .npy arrays in labeled folders.

    Args:
        info_types: e.g. ["Image:CLIP-ViT-B32", "Text:SentenceVec", "Voice:Whisper"]
        num_examples: int (applied to all) or list[int] (one per info_type)
        same_latent_space: True => unify all to max latent dim; False => keep native dims
        out_dir: root directory to write into (will be created if needed)
        write_readme: also write a short README.txt into the combo folder

    Returns:
        saved_paths: list of file paths written (in the order of info_types)
        manifest: dict describing encoders, dims, counts, and paths
    """
    # Normalize num_examples
    if isinstance(num_examples, int):
        num_examples = [num_examples] * len(info_types)
    elif isinstance(num_examples, list):
        if len(num_examples) != len(info_types):
            raise ValueError("Length of num_examples must match length of info_types.")
        if not all(isinstance(x, int) and x >= 0 for x in num_examples):
            raise ValueError("All entries in num_examples must be non-negative integers.")
    else:
        raise TypeError("num_examples must be int or list[int].")

    # Validate encoders
    for t in info_types:
        if t not in ENCODERS:
            raise KeyError(f"Unknown info type '{t}'. Known keys: {list(ENCODERS.keys())}")

    # Determine dims
    native_dims = [ENCODERS[t]["latent_dim"] for t in info_types]
    if same_latent_space:
        target_dim = max(native_dims)
        dims = [target_dim] * len(info_types)
        notes = ["Shared latent space"] * len(info_types)
    else:
        dims = native_dims
        notes = ["Separate latent space"] * len(info_types)

    # Create directories
    combo = _combo_label(info_types)                   # e.g., Image-CLIP-ViT-B32__Text-SentenceVec
    space = _space_label(same_latent_space, native_dims)
    combo_folder = os.path.join(out_dir, f"{combo}__{space}")
    os.makedirs(combo_folder, exist_ok=True)

    # Generate and save
    saved_paths = []
    for t, n, d in zip(info_types, num_examples, dims):
        arr = np.random.randn(n, d).astype(np.float32)
        filename = _file_label(t, n, d)
        fpath = os.path.join(combo_folder, filename)
        np.save(fpath, arr)
        saved_paths.append(fpath)

    # Compose manifest
    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "combo": info_types,
        "combo_label": combo,
        "latent_space": {
            "shared": bool(same_latent_space),
            "label": space,
            "target_dim_if_shared": max(native_dims) if same_latent_space else None
        },
        "items": []
    }

    for t, n, native_d, saved_d, note, path in zip(
        info_types, num_examples, native_dims, dims, notes, saved_paths
    ):
        manifest["items"].append({
            "info_type": t,
            "encoder_name": ENCODERS[t]["encoder"],
            "native_latent_dim": native_d,
            "saved_latent_dim": saved_d,
            "num_examples": n,
            "note": note,
            "path": path
        })

    # Write manifest + optional README
    manifest_path = os.path.join(combo_folder, "MANIFEST.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if write_readme:
        readme_path = os.path.join(combo_folder, "README.txt")
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(
                f"Toy Embedding Dataset\n"
                f"=====================\n"
                f"Combo: {combo}\n"
                f"Latent space: {'shared' if same_latent_space else 'separate'}\n"
                f"Folder: {combo_folder}\n\n"
                f"Files:\n"
            )
            for item in manifest["items"]:
                f.write(
                    f"- {os.path.basename(item['path'])} | "
                    f"{item['info_type']} | enc='{item['encoder_name']}' | "
                    f"n={item['num_examples']} | dim={item['saved_latent_dim']} "
                    f"({'native ' + str(item['native_latent_dim']) if item['saved_latent_dim'] != item['native_latent_dim'] else 'native'})\n"
                )

    return saved_paths, manifest

# --------------------------
# Example usage
# --------------------------
if __name__ == "__main__":
    info_types = ["Image:CLIP-ViT-B32", "Text:SentenceVec", "Voice:Whisper"]
    num_examples = [1_000, 10_000, 500]    # per-modality counts
    same_latent_space = True               # or False
    out_dir = "./toy_datasets"

    paths, manifest = generate_toy_dataset_to_disk(
        info_types=info_types,
        num_examples=num_examples,
        same_latent_space=same_latent_space,
        out_dir=out_dir
    )

    print("Saved files:")
    for p in paths:
        print(" -", p)

    print("\nEncoders & spaces:")
    for item in manifest["items"]:
        print(
            f"{item['info_type']}: {item['encoder_name']} | "
            f"native_dim={item['native_latent_dim']} -> saved_dim={item['saved_latent_dim']} | "
            f"n={item['num_examples']}"
        )
