import argparse
import os

from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(
        description="Upload a trained DeReC checkpoint folder to Hugging Face Hub."
    )
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help="Path to checkpoint directory (e.g. saved_models/rawfc_classifier/run_xxx/epoch_3).",
    )
    parser.add_argument(
        "--repo_id",
        required=True,
        help="HF repository in the form username/repo-name.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN", ""),
        help="HF token. If omitted, HF_TOKEN env variable is used.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    if not args.token:
        raise ValueError("HF token is empty. Pass --token or set HF_TOKEN.")

    api = HfApi(token=args.token)
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)
    api.upload_folder(
        folder_path=args.checkpoint_dir,
        repo_id=args.repo_id,
        repo_type="model",
    )
    print(f"Uploaded checkpoint from {args.checkpoint_dir} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
