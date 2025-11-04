# All comments in this code block are in English.
set -euo pipefail

src="/home/hulabdl/CMRxRecon2025/val"
dst_val="/home/lyy/dataset/cmr25/val"
dst_test="/home/lyy/dataset/cmr25/test"

mkdir -p "$dst_val" "$dst_test"

cd "$src"

# Collect top-level entries: regular files and symlinks (skip subdirectories)
# Use NUL delimiter to be safe with special characters
mapfile -d '' entries < <(find . -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -print0)

# Shuffle robustly (GNU shuf with -z for NUL entries)
mapfile -d '' shuffled < <(printf '%s\0' "${entries[@]}" | shuf -z)

n=${#shuffled[@]}
half=$(( n / 2 ))

for ((i=0; i<n; i++)); do
  f="${shuffled[$i]}"
  rel="${f#./}"
  abs="$src/$rel"

  # Resolve symlink to its ultimate target if the source is a symlink
  link_src="$abs"
  if [ -L "$abs" ]; then
    # readlink -f: canonical path; if broken, skip
    tgt="$(readlink -f -- "$abs" || true)"
    if [ -z "${tgt}" ] || [ ! -e "${tgt}" ]; then
      echo "Warning: skip broken symlink: $abs" >&2
      continue
    fi
    if [ -d "${tgt}" ]; then
      echo "Warning: skip symlink to directory: $abs -> $tgt" >&2
      continue
    fi
    link_src="${tgt}"
  fi

  if (( i < half )); then
    ln -sf -- "$link_src" "$dst_val/"
  else
    ln -sf -- "$link_src" "$dst_test/"
  fi
done

echo "Done. Total entries: $n; val: $half; test: $((n-half))"