#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

if ! env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  git -c http.proxy= -c https.proxy= submodule update --init --recursive \
  third_party/omniglue third_party/LineTR; then
  printf '%s\n' 'Direct GitHub access failed; retrying with the current network settings.' >&2
  git submodule update --init --recursive third_party/omniglue third_party/LineTR
fi
python -m pip install --no-build-isolation -e '.[full]'
python -m pip install --no-build-isolation --no-deps -e third_party/omniglue

mkdir -p models

download() {
  local url="$1"
  local destination="$2"
  if ! curl --noproxy '*' --fail --location --retry 2 "$url" --output "$destination"; then
    printf '%s\n' 'Direct model download failed; retrying with the current network settings.' >&2
    curl --fail --location --retry 2 "$url" --output "$destination"
  fi
}

if [[ ! -d models/sp_v6 ]]; then
  archive_path="$(mktemp --suffix=.tgz)"
  download https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz "$archive_path"
  tar -xzf "$archive_path" -C models
  rm -f "$archive_path"
fi

if [[ ! -f models/dinov2_vitb14_pretrain.pth ]]; then
  download https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth models/dinov2_vitb14_pretrain.pth
fi

if [[ ! -d models/og_export ]]; then
  archive_path="$(mktemp --suffix=.zip)"
  download https://storage.googleapis.com/omniglue/og_export.zip "$archive_path"
  unzip -q "$archive_path" -d models
  rm -f "$archive_path"
fi

echo "OmniGlue and LineTR models are ready."
