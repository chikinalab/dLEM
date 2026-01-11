from typer.testing import CliRunner
from dlem.cli import dlem_cli
import os

runner = CliRunner()


def test_dlem_drem_small(downloaded_data_dir):
# Minimal test of DLEM using Drosophila data
    out_path = os.path.join(os.getcwd(),'tests','drosophila_test_small')
    in_data = os.path.join(downloaded_data_dir, 'GSE263229_GlambdaLmerged_fix.mcool')
    print("Running DLEM for small Drosophila test")
    
    result = runner.invoke(dlem_cli, ["--output-tracks",
                                      "--norm",
                                      "--device", "cpu",
                                      "-l", "12800",
                                      "-r", "chr3R:6552201-7088853",
                                      in_data, 
                                      out_path])
    assert result.exit_code == 0

def test_dlem_drem_medium(downloaded_data_dir):
# Medium test of DLEM using Drosophila data
    out_path = os.path.join(os.getcwd(),'tests','drosophila_test_medium')
    in_bed = os.path.join(os.getcwd(),'tests', 'in_regions.bed')
    in_data = os.path.join(downloaded_data_dir, 'GSE263229_GlambdaLmerged_fix.mcool')
    print("Running DLEM against Drosophila regions in bed")
    print(f"{out_path} {in_data} {in_bed}")
    
    result = runner.invoke(dlem_cli, ["--output-tracks",
                                      "--plot",
                                      "--norm",
                                      "--device", "cpu",
                                      "-l", "3200",
                                      "-b", in_bed,
                                      in_data, 
                                      out_path])
    assert result.exit_code == 0

def test_dlem_drem_large(downloaded_data_dir):
# Large test of DLEM using Drosophila data
    out_path = os.path.join(os.getcwd(),'tests','drosophila_test_large')
    in_data = os.path.join(downloaded_data_dir, 'GSE263229_GlambdaLmerged_fix.mcool')
    in_bed = os.path.join(os.getcwd(),'tests', 'in_regions.bed')
    print("Running DLEM against Drosophila regions in bed")
    print(f"{out_path} {in_data} {in_bed}")
    
    result = runner.invoke(dlem_cli, ["--output-tracks",
                                      "--output-cool",
                                      "--plot",
                                      "--norm",
                                #     "--full-output",
                                      "--device", "cpu",
                                      "-l", "400",
                                      "-b", in_bed,
                                      "--all",
                                      in_data, 
                                      out_path])
    assert result.exit_code == 0