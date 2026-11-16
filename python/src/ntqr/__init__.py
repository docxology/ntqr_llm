"""
Evaluate noisy decision makers using logic and algebra.

Classes:

Functions:

Misc variables:

    __version__
    uci_adult_test_example
"""

import shutil
import os
import sys
from importlib import resources

__version__ = "0.9"


from ntqr.r2.examples import uciadult_label_counts

from ntqr.r2.datasketches import TrioLabelVoteCounts, TrioVoteCounts

from ntqr.r2.evaluators import (
    ErrorIndependentEvaluation,
    MajorityVotingEvaluation,
    SupervisedEvaluation,
)

from ntqr.labels import Label, Labels


def copy_notebooks() -> int:
    """Copy tutorials without overwriting an existing user destination."""
    try:
        source_path = resources.files("ntqr").joinpath("notebooks")
        destination = os.path.join(os.getcwd(), "ntqr_notebooks")
        shutil.copytree(str(source_path), destination)
    except OSError as error:
        print(f"Unable to copy notebooks: {error}", file=sys.stderr)
        return 1
    print(f"Notebooks copied to: {destination}")
    return 0
