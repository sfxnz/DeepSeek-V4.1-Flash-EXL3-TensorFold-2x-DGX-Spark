"""Hub listing of sfxnz/DeepSeek-V4.1-Flash-EXL3 at the pinned revision, and the recipe pins derived from it.
Read-only: model_info with file metadata, and hf_hub_download of config.json and the index into a scratch dir."""
import hashlib, json, sys, tempfile
from datetime import datetime, timezone
from huggingface_hub import HfApi, hf_hub_download, __version__

REPO, REV = "sfxnz/DeepSeek-V4.1-Flash-EXL3", "982b70452f399814f56b46272fd30394ae10d58c"
out = sys.argv[1]
info = HfApi().model_info(REPO, revision=REV, files_metadata=True)
with open(f"{out}/hub_files.tsv", "w") as fh:
    fh.write("path\tbytes\tlfs_sha256\tgit_blob_id\n")
    for s in sorted(info.siblings, key=lambda s: s.rfilename):
        fh.write(f"{s.rfilename}\t{s.size}\t{s.lfs.sha256 if s.lfs else ''}\t{s.blob_id or ''}\n")
tmp = tempfile.mkdtemp()
sha = {}
for name in ("config.json", "model.safetensors.index.json"):
    p = hf_hub_download(REPO, name, revision=REV, cache_dir=tmp)
    sha[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    idx = p if name.endswith("index.json") else None
shards = sorted(set(json.load(open(idx))["weight_map"].values()))
sizes = {s.rfilename: s.size for s in info.siblings}
pins = {
    "repo": REPO, "revision": REV, "hub_sha": info.sha, "files": len(info.siblings),
    "listed_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"), "huggingface_hub": __version__,
    "config_sha256": sha["config.json"], "index_sha256": sha["model.safetensors.index.json"],
    "shards": len(shards), "shard_bytes": sum(sizes[s] for s in shards),
    "all_files_bytes": sum(sizes.values()), "shards_missing_from_listing": [s for s in shards if s not in sizes],
}
json.dump(pins, open(f"{out}/pins.json", "w"), indent=1)
open(f"{out}/pins.json", "a").write("\n")
print(json.dumps(pins, indent=1))
