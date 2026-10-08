#!/bin/bash
set -e

REPO_ROOT="/share/klab/labstudents/oshtyria/revision-ContrastiveCoding"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Step 1: downloading CompSearch.zip from Konkle Lab ==="
curl -fL -o "$SCRIPT_DIR/compsearch.zip" "https://konklab.fas.harvard.edu/ImageSets/CompSearch.zip"
echo "Downloaded: $(ls -la "$SCRIPT_DIR/compsearch.zip")"

echo "=== Step 2: unzipping ==="
rm -rf "$SCRIPT_DIR/compsearch_extract"
mkdir -p "$SCRIPT_DIR/compsearch_extract"
unzip -q "$SCRIPT_DIR/compsearch.zip" -d "$SCRIPT_DIR/compsearch_extract"

echo "=== Step 3: sanity check - confirming SHINE-processed (grayscale, matched luminance) ==="
python3 -c "
from PIL import Image
import numpy as np
import os

base = '$SCRIPT_DIR/compsearch_extract/CompSearch'
means = []
for categ in ['faces', 'bodies', 'cats', 'buildings', 'cars', 'chairs', 'hammers', 'phones']:
    d = os.path.join(base, categ)
    files = sorted([f for f in os.listdir(d) if not f.startswith('.')])[:3]
    for f in files:
        img = Image.open(os.path.join(d, f))
        arr = np.array(img.convert('L'))
        means.append(arr.mean())
        print(f'{categ}/{f}: mode={img.mode}, size={img.size}, mean={arr.mean():.2f}, std={arr.std():.2f}')

spread = max(means) - min(means)
print(f'\nLuminance spread across all samples: {spread:.2f}')
if spread > 5:
    print('WARNING: luminance not tightly matched - this may not be the SHINE-processed set!')
else:
    print('OK: luminance tightly matched across categories - consistent with SHINE processing')
"

echo ""
echo "=== Step 4: backing up any existing mc8-shined ==="
if [ -d "$REPO_ROOT/stimulus_sets/mc8-shined" ]; then
    rm -rf "$REPO_ROOT/stimulus_sets/mc8-shined-corrupted-backup"
    mv "$REPO_ROOT/stimulus_sets/mc8-shined" "$REPO_ROOT/stimulus_sets/mc8-shined-corrupted-backup"
    echo "Old mc8-shined backed up to mc8-shined-corrupted-backup"
fi

echo "=== Step 5: building mc8-shined with correctly numbered folder names ==="
mkdir -p "$REPO_ROOT/stimulus_sets/mc8-shined"

declare -A MAP=(
    [faces]="1-faces"
    [bodies]="2-bodies"
    [cats]="3-cats"
    [buildings]="4-buildings"
    [cars]="5-cars"
    [chairs]="6-chairs"
    [hammers]="7-hammers"
    [phones]="8-phones"
)

for src_name in "${!MAP[@]}"; do
    dest_name="${MAP[$src_name]}"
    mkdir -p "$REPO_ROOT/stimulus_sets/mc8-shined/$dest_name"
    find "$SCRIPT_DIR/compsearch_extract/CompSearch/$src_name" -maxdepth 1 -type f ! -name ".*" \
        -exec cp {} "$REPO_ROOT/stimulus_sets/mc8-shined/$dest_name/" \;
done

echo "=== Step 6: final verification ==="
TOTAL_EMPTY=0
for d in "$REPO_ROOT"/stimulus_sets/mc8-shined/*/; do
    categ=$(basename "$d")
    total=$(find "$d" -maxdepth 1 -type f | wc -l)
    empty=$(find "$d" -maxdepth 1 -type f -size 0 | wc -l)
    echo "$categ: $total files, $empty empty"
    TOTAL_EMPTY=$((TOTAL_EMPTY + empty))
done

echo "=== Step 7: cleaning up temporary files ==="
rm -rf "$SCRIPT_DIR/compsearch.zip" "$SCRIPT_DIR/compsearch_extract"

if [ "$TOTAL_EMPTY" -eq 0 ]; then
    echo ""
    echo "=== SUCCESS: mc8-shined rebuilt with 0 empty files ==="
else
    echo ""
    echo "=== WARNING: $TOTAL_EMPTY empty files remain - something went wrong ==="
fi
