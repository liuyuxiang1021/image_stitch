#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repository_root"

git submodule update --init --recursive
python -m pip install --no-build-isolation -e '.[full]'
python -m pip install --no-build-isolation --no-deps -e third_party/omniglue

mkdir -p models

if [[ ! -d models/sp_v6 ]]; then
  archive_path="$(mktemp --suffix=.tgz)"
  curl --fail --location \
    https://github.com/rpautrat/SuperPoint/raw/master/pretrained_models/sp_v6.tgz \
    --output "$archive_path"
  tar -xzf "$archive_path" -C models
  rm -f "$archive_path"
fi

if [[ ! -f models/dinov2_vitb14_pretrain.pth ]]; then
  curl --fail --location \
    https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth \
    --output models/dinov2_vitb14_pretrain.pth
fi

if [[ ! -d models/og_export ]]; then
  archive_path="$(mktemp --suffix=.zip)"
  curl --fail --location \
    https://storage.googleapis.com/omniglue/og_export.zip \
    --output "$archive_path"
  unzip -q "$archive_path" -d models
  rm -f "$archive_path"
fi

echo "OmniGlue and LineTR models are ready."
