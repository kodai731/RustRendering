"""Copy the ZIPs already served by the live extension repository into a site
directory so a redeploy of one extension keeps the others published.

Entries whose ``id`` matches ``--exclude-id`` are skipped: those are being
replaced by the ZIPs the caller has just built into the same directory.
"""
import argparse
import json
import urllib.request
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--site", required=True, help="directory receiving the ZIPs")
    parser.add_argument("--url", required=True, help="live Pages base URL (contains index.json)")
    parser.add_argument("--exclude-id", action="append", default=[], help="extension id being republished")
    return parser.parse_args()


def fetch_index(base_url):
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/index.json", timeout=30) as response:
            return json.load(response)
    except Exception as error:
        print(f"[merge_live_extensions] no live index ({error}); nothing to keep")
        return {"data": []}


def main():
    args = parse_args()
    site = Path(args.site)
    site.mkdir(parents=True, exist_ok=True)
    kept = 0
    for entry in fetch_index(args.url).get("data", []):
        if entry.get("id") in args.exclude_id:
            continue
        name = Path(entry["archive_url"]).name
        target = site / name
        if target.exists():
            continue
        urllib.request.urlretrieve(f"{args.url.rstrip('/')}/{name}", target)
        kept += 1
        print(f"[merge_live_extensions] kept {entry['id']} {entry.get('version')} {name}")
    print(f"[merge_live_extensions] {kept} ZIPs kept from the live site")


main()
