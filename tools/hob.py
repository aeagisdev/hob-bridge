#!/usr/bin/env python3
"""hob — Hob Bridge command-line helpers.

    python tools/hob.py sync ./models               # upload what's new or changed
    python tools/hob.py sync ./models --dry-run     # only report the diff
    python tools/hob.py sync ./models --only-new    # never overwrite files on the volume

`sync` compares a local models folder with the network volume through RunPod's
S3-compatible API — by relative path and size — and uploads the difference to
`models/` on the volume (the layout the worker mounts). Files that exist only on
the volume are reported, never deleted.

Configuration (environment; secrets are never taken as arguments):
    HOB_VOLUME_ID            network volume id (the S3 bucket)
    HOB_VOLUME_DATACENTER    its datacenter, e.g. EU-RO-1 (the S3 region)
    HOB_VOLUME_KEY_ID        S3 API key access id (user_...)  — RunPod console →
    HOB_VOLUME_SECRET        S3 API key secret (rps_...)        Credentials → S3 API Keys
    HOB_PART_MB              upload part size in MB (default 32; lower it if parts time out)
Needs boto3 (pip install boto3).
"""
import argparse
import os
import sys
import time

# RunPod's S3 API caps a single PutObject / UploadPart at 500 MB, but its proxy answers
# 524 to a part that takes too long to store: 256 MB parts failed on a real volume,
# so parts are small and few at a time. ponytail: 32 MB parts cap a file at ~320 GB
# (S3's 10,000 parts); raise HOB_PART_MB for larger files.
PART_SIZE = int(os.environ.get("HOB_PART_MB", "32")) * 1024 * 1024
MAX_CONCURRENCY = 4
SKIP_SUFFIXES = (".tmp", ".part", ".crdownload")


def scan_local(root):
    """{relative posix path: size} for the files under root (dotfiles and partial downloads skipped).

    Symlinked folders are followed, so e.g. models/embeddings can point at A1111's embeddings/.
    """
    files = {}
    for folder, dirs, names in os.walk(root, followlinks=True):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for name in names:
            if name.startswith(".") or name.endswith(SKIP_SUFFIXES):
                continue
            path = os.path.join(folder, name)
            files[os.path.relpath(path, root).replace(os.sep, "/")] = os.path.getsize(path)
    return files


def scan_remote(client, bucket, prefix, attempts=3, pause=10):
    """{relative path: size} under prefix/ on the volume.

    RunPod computes ETags for files written outside the S3 API on the first
    listing, which can fail ("same next token received twice"): retried.
    """
    for attempt in range(attempts):
        try:
            files = {}
            pages = client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + "/")
            for page in pages:
                for obj in page.get("Contents", []):
                    if not obj["Key"].endswith("/"):
                        files[obj["Key"][len(prefix) + 1:]] = obj["Size"]
            return files
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            print(f"listing the volume failed ({exc}); retrying in {pause}s", file=sys.stderr)
            time.sleep(pause)


def plan(local, remote):
    """(new, changed, remote_only): sorted relative paths."""
    new = sorted(path for path in local if path not in remote)
    changed = sorted(path for path in local if path in remote and remote[path] != local[path])
    remote_only = sorted(path for path in remote if path not in local)
    return new, changed, remote_only


def human(size):
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def report(local, remote, new, changed, remote_only, only_new):
    """Print what differs, grouped by model folder (Lora/, VAE/, …)."""
    def by_folder(paths, sizes):
        groups = {}
        for path in paths:
            folder = path.split("/", 1)[0] if "/" in path else "."
            count, total = groups.get(folder, (0, 0))
            groups[folder] = (count + 1, total + sizes[path])
        return groups

    sections = [("missing on the volume", new, local),
                ("different size on the volume" + (" (kept: --only-new)" if only_new else ""), changed, local),
                ("only on the volume (never deleted)", remote_only, remote)]
    for title, paths, sizes in sections:
        if not paths:
            continue
        print(f"{title}: {len(paths)} files, {human(sum(sizes[p] for p in paths))}")
        for folder, (count, total) in sorted(by_folder(paths, sizes).items()):
            print(f"  {folder}/: {count} files, {human(total)}")
        for path in paths:
            print(f"    {path} ({human(sizes[path])})")
    if not (new or changed or remote_only):
        print("in sync: nothing to do")


def upload(client, bucket, prefix, root, paths, sizes, transfer_config=None):
    for index, path in enumerate(paths, 1):
        size, sent, started = sizes[path], [0], time.time()

        def progress(chunk, path=path, size=size, sent=sent):
            sent[0] += chunk
            print(f"\r[{index}/{len(paths)}] {path} {sent[0] * 100 // max(size, 1)}%", end="", flush=True)

        extra = {"Config": transfer_config} if transfer_config else {}
        client.upload_file(os.path.join(root, *path.split("/")), bucket, f"{prefix}/{path}",
                           Callback=progress, **extra)
        print(f"\r[{index}/{len(paths)}] {path} done ({human(size)}, {time.time() - started:.0f}s)")


def sync(client, bucket, root, prefix="models", dry_run=False, only_new=False, transfer_config=None):
    local = scan_local(root)
    if not local:
        print(f"no files under {root}", file=sys.stderr)
        return 1
    remote = scan_remote(client, bucket, prefix)
    new, changed, remote_only = plan(local, remote)
    report(local, remote, new, changed, remote_only, only_new)
    todo = new + ([] if only_new else changed)
    if dry_run or not todo:
        return 0
    upload(client, bucket, prefix, root, todo, local, transfer_config)
    return 0


def _client():
    try:
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config
    except ImportError:
        sys.exit("hob sync needs boto3: pip install boto3")
    datacenter = os.environ.get("HOB_VOLUME_DATACENTER", "").strip()
    if not datacenter or not os.environ.get("HOB_VOLUME_ID", "").strip():
        sys.exit("set HOB_VOLUME_ID and HOB_VOLUME_DATACENTER (e.g. EU-RO-1)")
    client = boto3.client(
        "s3",
        endpoint_url=f"https://s3api-{datacenter.lower()}.runpod.io",
        region_name=datacenter,
        # None falls back to boto3's usual credential chain (AWS_* env, ~/.aws)
        aws_access_key_id=os.environ.get("HOB_VOLUME_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("HOB_VOLUME_SECRET") or None,
        # read_timeout: RunPod's advice for multi-GB multipart completions
        config=Config(read_timeout=7200, retries={"max_attempts": 5, "mode": "standard"},
                      s3={"addressing_style": "path"}),
    )
    return client, TransferConfig(multipart_threshold=PART_SIZE, multipart_chunksize=PART_SIZE,
                                  max_concurrency=MAX_CONCURRENCY)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="hob", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    sync_cmd = commands.add_parser("sync", help="diff a local models folder against the volume and upload")
    sync_cmd.add_argument("local", help="local folder with the volume layout (Stable-diffusion/, Lora/, …)")
    sync_cmd.add_argument("--prefix", default="models", help="folder on the volume (default: models)")
    sync_cmd.add_argument("--dry-run", action="store_true", help="report only, upload nothing")
    sync_cmd.add_argument("--only-new", action="store_true", help="upload missing files only, never overwrite")
    args = parser.parse_args(argv)

    client, transfer_config = _client()
    return sync(client, os.environ["HOB_VOLUME_ID"].strip(), args.local, args.prefix.strip("/"),
                args.dry_run, args.only_new, transfer_config)


if __name__ == "__main__":
    sys.exit(main())
