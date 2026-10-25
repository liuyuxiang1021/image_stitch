#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

# Try direct GitHub and Dropbox access first. Fall back to the caller's
# configured network only when this server cannot reach the upstream directly.
if ! env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  git -c http.proxy= -c https.proxy= submodule update --init third_party/RAFT; then
  printf '%s\n' 'Direct GitHub access failed; retrying with the current network settings.' >&2
  git submodule update --init third_party/RAFT
fi

mkdir -p models
if [[ -f models/raft-things.pth ]]; then
  printf '%s\n' 'RAFT Things checkpoint already exists.'
  exit 0
fi

archive_path="$(mktemp --suffix=.zip)"
trap 'rm -f "$archive_path"' EXIT
model_url='https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip'
if ! curl --noproxy '*' --fail --location --retry 2 "$model_url" --output "$archive_path"; then
  printf '%s\n' 'Direct checkpoint download failed; retrying with the current network settings.' >&2
  curl --fail --location --retry 2 "$model_url" --output "$archive_path"
fi

checkpoint_in_zip="$(unzip -Z1 "$archive_path" | awk '/(^|\/)raft-things\.pth$/ {print; exit}')"
if [[ -z "$checkpoint_in_zip" ]]; then
  printf '%s\n' 'The official RAFT archive does not contain raft-things.pth.' >&2
  exit 1
fi
unzip -p "$archive_path" "$checkpoint_in_zip" > models/raft-things.pth
printf 'RAFT checkpoint saved to %s\n' "$repository_root/models/raft-things.pth"
