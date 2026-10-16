#!/bin/bash
# Phase 7.2: download the 4 SACSoN (HuRoN) folders the confirmation block names, resume-capable (curl -C -).
#
#   SACSON_RAW=/mnt/shared_disk/nazli/nomad_data/sacson_raw mapmad/scripts/p7_download_sacson.sh
#
# Writes <SACSON_RAW>/<folder>/<bag>, a file list with sizes (sizes.tsv) and sha256 sums (SHA256SUMS).
# Re-running skips finished files (their size matches the server's Content-Length) and resumes partial ones.
set -euo pipefail

BASE_URL="${SACSON_URL:-https://rail.eecs.berkeley.edu/datasets/huron}"
DEST="${SACSON_RAW:?set SACSON_RAW to the download folder}"
FOLDERS=(Dec-06-2022-bww8 Jan-12-2023-bww8 Feb-15-2023-cory1 Feb-17-2023-soda3)

mkdir -p "$DEST"
: > "$DEST/sizes.tsv.tmp"
for folder in "${FOLDERS[@]}"; do
    mkdir -p "$DEST/$folder"
    bags=$(curl -sfL "$BASE_URL/$folder/" | grep -o 'href="[^"?/]*\.bag"' | sed 's/href="//; s/"$//' | sort -u)
    for bag in $bags; do
        url="$BASE_URL/$folder/$bag"
        out="$DEST/$folder/$bag"
        remote=$(curl -sfIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:" {n=$2} END {print n}')
        local_size=$(stat -c %s "$out" 2>/dev/null || echo 0)
        if [ "$local_size" != "$remote" ]; then
            echo "get $folder/$bag ($remote bytes, have $local_size)"
            curl -sfL --retry 5 --retry-delay 10 -C - -o "$out" "$url"
        fi
        local_size=$(stat -c %s "$out")
        [ "$local_size" = "$remote" ] || { echo "size mismatch: $out $local_size != $remote" >&2; exit 1; }
        printf '%s/%s\t%s\n' "$folder" "$bag" "$local_size" >> "$DEST/sizes.tsv.tmp"
    done
done
mv "$DEST/sizes.tsv.tmp" "$DEST/sizes.tsv"
(cd "$DEST" && sha256sum */*.bag > SHA256SUMS)
awk -F'\t' '{split($1, p, "/"); n[p[1]]++; s[p[1]] += $2; t += $2}
    END {for (f in n) printf "%s: %d bags, %.2f GB\n", f, n[f], s[f] / 1e9; printf "total: %.2f GB\n", t / 1e9}' "$DEST/sizes.tsv"
