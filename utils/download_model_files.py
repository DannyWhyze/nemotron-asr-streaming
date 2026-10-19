"""
Downloadt gezielt einzelne Dateien aus einem Hugging-Face-Repo (nicht das
komplette Repo) nach <models_dir>/<repo--id> und protokolliert
Revision + Lizenz in download_info.json.

Beispiel:
    uv run python utils/download_model_files.py unsloth/gemma-4-E4B-it-GGUF "gemma-4-E4B-it-UD-Q8_K_XL.gguf" "mmproj-F32.gguf"
"""

import json
import os
import sys
from datetime import UTC, datetime

from dotenv import load_dotenv
from huggingface_hub import HfApi, snapshot_download

load_dotenv()
DEFAULT_CACHE_DIR = os.getenv("HF_HOME") or os.getenv("MODELS_DIR") or os.path.expanduser("~/.cache/huggingface")


def write_download_info(target_dir, model_id, token):
    """Protokolliert Repo-ID, Revision und Lizenz — wie im Haupt-Downloader."""
    info = {
        "repo_id": model_id,
        "downloaded_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "revision": None,
        "license": None,
        "last_modified": None,
    }
    try:
        meta = HfApi().model_info(repo_id=model_id, token=token)
        info["revision"] = meta.sha
        info["last_modified"] = str(getattr(meta, "last_modified", None))
        card = getattr(meta, "card_data", None)
        if card is not None:
            info["license"] = getattr(card, "license", None) or (card.get("license") if hasattr(card, "get") else None)
    except Exception as e:  # noqa: BLE001 - bookkeeping must never break the download
        print(f"Hinweis: Revision konnte nicht ermittelt werden ({e})")

    path = os.path.join(target_dir, "download_info.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2, ensure_ascii=False)
    print(f"Revision:  {info['revision']}")
    print(f"Lizenz:    {info['license']}")
    print(f"Protokoll: {path}")


def download_model_files(model_id, patterns, cache_dir=DEFAULT_CACHE_DIR):
    """Laedt nur die Dateien, die auf die fnmatch-Muster passen."""
    print("\n--- Hugging Face Selective File Downloader ---")
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token or token == "DEIN_TOKEN_HIER_EINFUEGEN":
        print("Hinweis: Kein HF_TOKEN in der .env gefunden. Oeffentliche Modelle laden trotzdem.")
    else:
        print("Authentifizierung: HF_TOKEN geladen.")

    print(f"Repo:    {model_id}")
    print(f"Muster:  {patterns}")
    target_dir = os.path.join(cache_dir, model_id.replace("/", "--"))
    os.makedirs(target_dir, exist_ok=True)
    print(f"Ziel:    {target_dir}")

    try:
        local_path = snapshot_download(
            repo_id=model_id,
            local_dir=target_dir,
            token=token,
            max_workers=8,
            allow_patterns=patterns,
        )
        print("\nErfolg! Die Dateien liegen hier:")
        print(local_path)
        write_download_info(target_dir, model_id, token)
        return local_path
    except Exception as e:  # noqa: BLE001 - report download failure instead of raising
        print(f"\nFehler beim Download: {e}")
        return None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Aufruf: python utils/download_model_files.py <repo_id> [datei_muster ...]")
    repo = sys.argv[1]
    patterns = sys.argv[2:] or None
    if not patterns:
        sys.exit("Bitte mindestens ein Dateimuster angeben (sonst: download_model.py nutzen).")
    download_model_files(repo, patterns)
