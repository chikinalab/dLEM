import pytest
import requests
import sh

@pytest.fixture(scope="session")
def downloaded_data_dir(tmp_path_factory):
    """
    Downloads input test chromatin looping data, preps it using hictk
    """
    temp_dir = tmp_path_factory.mktemp("test_data")
    data_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE263nnn/GSE263229/suppl/GSE263229%5FGlambdaLmerged.mcool"
    drem_path = temp_dir / "GSE263229_GlambdaLmerged.mcool"
    drem_path_fix = temp_dir / "GSE263229_GlambdaLmerged_fix.mcool"

    print(f"\nDownloading data from {data_url} to {drem_path}...")

    response = requests.get(data_url, stream=True)
    response.raise_for_status()
    with open(drem_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Downloaded test input chromatin looping data, now fixin mcool")

    sh.hictk("fix-mcool","-t", "8", drem_path, drem_path_fix)

    yield temp_dir
