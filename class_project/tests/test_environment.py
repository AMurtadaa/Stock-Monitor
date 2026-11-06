"""Reproduce Colab's inherited notebook backend in a fresh child interpreter."""
import os
import subprocess
import sys

from class_project.common import ROOT


def test_colab_backend_is_replaced_before_matplotlib_import():
    environment = dict(os.environ, MPLBACKEND="module://matplotlib_inline.backend_inline")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import class_project.common; import matplotlib; "
            "assert matplotlib.get_backend().lower() == 'agg'; "
            "from matplotlib.figure import Figure; "
            "import io; b=io.BytesIO(); Figure().savefig(b, format='png'); "
            "assert b.getvalue().startswith(b'\\x89PNG'); print('backend and PNG passed')",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "backend and PNG passed" in result.stdout
