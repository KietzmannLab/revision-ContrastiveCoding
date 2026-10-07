#!/bin/bash
set -e

cd /share/klab/labstudents/oshtyria/revision-ContrastiveCoding

echo "=== Step 1: downloading fresh copy from Dataverse ==="
curl -fL -o dataverse_fresh.zip "https://dataverse.harvard.edu/api/access/datafile/10260721"
echo "Downloaded: $(ls -la dataverse_fresh.zip)"

echo "=== Step 2: outer unzip ==="
rm -rf dataverse_fresh_extract
mkdir -p dataverse_fresh_extract
unzip -q dataverse_fresh.zip -d dataverse_fresh_extract

echo "=== Step 3: locating inner classic-categ.zip ==="
INNER_ZIP=$(find dataverse_fresh_extract -iname "classic-categ.zip" | head -1)
if [ -z "$INNER_ZIP" ]; then
    echo "ERROR: could not find classic-categ.zip inside the outer archive"
    exit 1
fi
echo "Found: $INNER_ZIP"

echo "=== Step 4: inner unzip ==="
rm -rf classic_categ_inner_extract
mkdir -p classic_categ_inner_extract
unzip -q "$INNER_ZIP" -d classic_categ_inner_extract

echo "=== Step 5: rebuilding stimulus_sets/classic-categ from scratch, non-empty files only ==="
rm -rf stimulus_sets/classic-categ
mkdir -p stimulus_sets/classic-categ

for cat in 1-Faces 2-Bodies 3-Scenes 4-Words 5-Objects 6-ScrambledObjects 7-ScrambledWords; do
    mkdir -p "stimulus_sets/classic-categ/$cat"
done

COPIED=0
SKIPPED_EMPTY=0
while IFS= read -r -d '' f; do
    categ_dir=$(basename "$(dirname "$f")")
    fname=$(basename "$f")
    case "$categ_dir" in
        1-Faces|2-Bodies|3-Scenes|4-Words|5-Objects|6-ScrambledObjects|7-ScrambledWords)
            if [ -s "$f" ]; then
                cp "$f" "stimulus_sets/classic-categ/$categ_dir/$fname"
                COPIED=$((COPIED+1))
            else
                SKIPPED_EMPTY=$((SKIPPED_EMPTY+1))
            fi
            ;;
    esac
done < <(find classic_categ_inner_extract -type f ! -name "._*" ! -name ".DS_Store" -print0)

echo "Copied: $COPIED non-empty files"
echo "Skipped: $SKIPPED_EMPTY empty files"

echo "=== Step 6: per-category verification ==="
for d in stimulus_sets/classic-categ/*/; do
    categ=$(basename "$d")
    total=$(find "$d" -maxdepth 1 -type f | wc -l)
    empty=$(find "$d" -maxdepth 1 -type f -size 0 | wc -l)
    echo "$categ: $total files, $empty empty"
done

echo "=== Cleaning up temporary files ==="
rm -rf dataverse_fresh.zip dataverse_fresh_extract classic_categ_inner_extract

echo "=== Done ==="
