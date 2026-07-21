#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GROUP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
THIRD_PARTY_DIR="${GROUP_DIR}/third_party"
VERSIONS_FILE="${GROUP_DIR}/third_party_versions.txt"

mkdir -p "${THIRD_PARTY_DIR}"

clone_repo() {
  local url="$1"
  local name="$2"
  local target="${THIRD_PARTY_DIR}/${name}"

  if [[ -d "${target}/.git" ]]; then
    echo "[skip] ${name} already exists"
  else
    echo "[clone] ${url}"
    git clone --depth 1 "${url}" "${target}"
  fi
}

clone_repo "https://github.com/moojink/openvla-oft.git" "openvla-oft"
clone_repo "https://github.com/Lifelong-Robot-Learning/LIBERO.git" "LIBERO"
clone_repo "https://github.com/VLA-Trace/VLA-Trace.git" "VLA-Trace"
clone_repo "https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas.git" "mechanistic-steering-vlas"

: > "${VERSIONS_FILE}"
for repo in openvla-oft LIBERO VLA-Trace mechanistic-steering-vlas; do
  sha="$(git -C "${THIRD_PARTY_DIR}/${repo}" rev-parse HEAD)"
  printf "%s %s\n" "${repo}" "${sha}" >> "${VERSIONS_FILE}"
done

echo "[done] repositories saved under ${THIRD_PARTY_DIR}"
echo "[done] versions written to ${VERSIONS_FILE}"
