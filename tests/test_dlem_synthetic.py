from typer.testing import CliRunner
import pandas as pd
import numpy as np
import cooler
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from dlem.cli import dlem_cli
import os

runner = CliRunner()

def synthetic_cool(out_name = "synthetic_data",
                   resolution = 1,
                   num_rand_ints = 2,
                   ct_mean = 5.0,
                   ct_sd = 1.0,
                   diag_vs_rest = 2.0,
                   seed = 24,
                   chr_size = 10
                   ):
    np.random.seed(seed)
    #Synthetic data parameters
    manual_ints1 = [100]
    manual_ints2 = [30]
    manual_counts = np.random.normal(ct_mean, ct_sd, size=len(manual_ints1))
    chromsizes = pd.Series({'chr1': chr_size})
    bins = cooler.binnify(chromsizes, resolution)
    bins['weight'] = 1.0
    n_bins = len(bins)
    
    # Diagonal interactions (strongest)
    bin1_ids = np.arange(n_bins)
    bin2_ids = np.arange(n_bins)
    counts = np.random.normal(ct_mean*diag_vs_rest, ct_sd,
                               size=n_bins)
    
    # Add some off-diagonal interactions
    off_diag_bin1 = np.random.randint(0, n_bins, size=num_rand_ints)
    off_diag_bin2 = np.random.randint(0, n_bins, size=num_rand_ints)
    nondiag = ~(off_diag_bin1 == off_diag_bin2)
    off_diag_counts = np.random.normal(ct_mean, ct_sd, size=num_rand_ints)
    off_diag_bin1 = off_diag_bin1[nondiag]
    off_diag_bin2 = off_diag_bin2[nondiag]
    off_diag_counts = off_diag_counts[nondiag]
    
    bin1_ids = np.concatenate([bin1_ids, off_diag_bin1, manual_ints1])
    bin2_ids = np.concatenate([bin2_ids, off_diag_bin2, manual_ints2])
    counts = np.concatenate([counts, off_diag_counts, manual_counts])
    mask = bin1_ids <= bin2_ids
    bin1_ids_sorted = bin1_ids[mask]
    bin2_ids_sorted = bin2_ids[mask]
    counts_sorted = counts[mask]
    
    pixels = pd.DataFrame({
        'bin1_id': bin1_ids_sorted,
        'bin2_id': bin2_ids_sorted,
        'count': counts_sorted
    })
    
    pixels.sort_values(['bin1_id', 'bin2_id'], inplace=True)
    pixels = pixels.drop_duplicates(subset=['bin1_id', 'bin2_id'])
    pixels.reset_index(drop=True, inplace=True)
    
    output_file = os.path.join(os.getcwd(),'tests',f'{out_name}.cool')
    cooler.create_cooler(output_file, bins, pixels, assembly='synthetic_assembly')
    
    
    cool_handle = cooler.Cooler(output_file)
    full_first_chr = f"{chromsizes.keys()[0]}:0-{chromsizes[chromsizes.keys()[0]]}"
    contact_map = cool_handle.matrix(balance=True).fetch(full_first_chr)
    contact_map[contact_map < 1] = 1
    norm_col = colors.LogNorm(vmin=contact_map.min(), vmax=contact_map.max())
    
    plt.imshow(contact_map, norm=norm_col)
    plt.colorbar()
    out_img = os.path.join(os.getcwd(),'tests',f'{out_name}.png')
    plt.savefig(out_img)
    plt.close()



def test_dlem_synthetic_minimal():
# Minimal test of DLEM using synthetic data
    out_name = "synthetic_data"
    synthetic_cool(out_name)
    out_path = os.path.join(os.getcwd(),'tests','test_synthetic_out')
    in_data = os.path.join(os.getcwd(),'tests', f'{out_name}.cool')
    print("Running synthetic data through dlem")
    result = runner.invoke(dlem_cli, ["--debug",
                                      "--plot",
                                      "-w", "5",
                                      "-r", "chr1:0-10",
                                      in_data, 
                                      out_path])
    assert result.exit_code == 0

def test_dlem_synthetic_medium():
# Minimal test of DLEM using synthetic data
    out_name = "synthetic_data_med"
    resolution = 100
    num_rand_ints = 30000
    chr_size = 100000
    ct_mean = 50.0
    
    synthetic_cool(out_name=out_name, 
                   resolution=resolution, 
                   num_rand_ints=num_rand_ints,
                   ct_mean=ct_mean,
                   chr_size=chr_size)
    out_path = os.path.join(os.getcwd(),'tests','test_out')
    in_data = os.path.join(os.getcwd(),'tests', f'{out_name}.cool')
    print("Running synthetic data through dlem")
    result = runner.invoke(dlem_cli, ["--debug",
                                      "--plot",
                                      "-r", "chr1:0-100000",
                                      in_data, 
                                      out_path])
    assert result.exit_code == 0
