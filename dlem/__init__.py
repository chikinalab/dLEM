__version__ = "0.1.0"


def _check_gpu():
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi'], capture_output=True, timeout=3
        )
        if result.returncode != 0:
            return  # no NVIDIA GPU
        import jax
        if jax.default_backend() == 'cpu':
            print(
                '[dlem] NVIDIA GPU detected but JAX is running on CPU.\n'
                '       For GPU acceleration reinstall with:\n'
                '           pip install "dlem-jax[cuda]"\n',
                flush=True,
            )
    except Exception:
        pass  # never crash on import due to this check


_check_gpu()
