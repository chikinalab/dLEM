"""
Creates docs/data/example_chr10.cool from the local H1-hESC Hi-C file.

Run once (requires the full mcool locally):
    python docs/data/create_example_data.py

Source : H1-hESC 10 kb Hi-C (4DNFI9GMP2J8), accessed via local copy of
         /home/tis97/data2/loop_extrusion_project/4DNucleome/H1hESC.mcool
Region : chr10:19–26 Mb (700 bins at 10 kb).
         The reference locus shown in the paper (chr10:20.5–22.5 Mb) falls at
         local bins 150–350 within this window.
Chromosome name: stored as 'ref_region' (coordinates shifted to start at 0;
         add 19,000,000 bp to recover true chr10 positions).
Normalization weights (ICE): kept as-is from the original file.
"""
import os
import numpy as np
import cooler

MCOOL  = '/home/tis97/data2/loop_extrusion_project/4DNucleome/H1hESC.mcool'
RES    = 10_000
REGION = 'chr10:19000000-26000000'
OUT    = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'example_chr10.cool')

clr = cooler.Cooler(f'{MCOOL}::resolutions/{RES}')

# Bins for the region, preserving original ICE balancing weights
bins_full = clr.bins().fetch(REGION)
bins = bins_full.reset_index(drop=True)
old_to_new = {old: new for new, old in enumerate(bins_full.index)}

# Shift coordinates to start at 0 (required by hictkpy's bin-offset index)
# and rename the chromosome so the file is self-contained and unambiguous.
OFFSET = int(bins['start'].iloc[0])
bins['chrom'] = 'ref_region'
bins['start'] = bins['start'] - OFFSET
bins['end']   = bins['end']   - OFFSET

# Pixels (raw counts) for the region, remapped to new bin IDs
pixels = clr.pixels(join=False).fetch(REGION).reset_index(drop=True)
pixels['bin1_id'] = pixels['bin1_id'].map(old_to_new)
pixels['bin2_id'] = pixels['bin2_id'].map(old_to_new)
pixels = (pixels
          .dropna(subset=['bin1_id', 'bin2_id'])
          .astype({'bin1_id': int, 'bin2_id': int}))

cooler.create_cooler(
    OUT,
    bins=bins[['chrom', 'start', 'end', 'weight']],
    pixels=pixels[['bin1_id', 'bin2_id', 'count']],
    dtypes={'count': np.float32},
    metadata={
        'source': '4DNFI9GMP2J8 H1-hESC',
        'true_region': REGION,
        'offset_bp': OFFSET,
        'resolution': RES,
    },
    ordered=True,
    symmetric_upper=True,
    triucheck=False,
    dupcheck=False,
)
size_mb = os.path.getsize(OUT) / 1e6
print(f'Written: {OUT}  ({size_mb:.1f} MB, {len(bins)} bins, {len(pixels)} pixels)')
