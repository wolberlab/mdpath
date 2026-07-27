"""Edge/residue helpers --- :mod:`mdpath.src.edge_utils`
===============================================================================

Small shared utilities for turning residue labels into canonical integer edge
keys. Centralising these avoids the previously duplicated ``_pair`` /
``_residue_id`` logic in :mod:`mdpath.src.confidence`,
:mod:`mdpath.src.null_model`, :mod:`mdpath.src.mutual_information` and
:mod:`mdpath.src.path_confidence_viz`.

Functions
---------

:func:`residue_id`

:func:`canonical_pair`
"""

from typing import Sequence, Tuple


def residue_id(label) -> int:
    """Return the numeric residue ID from either ``42`` or ``'Res 42'``.

    Args:
        label: A residue identifier -- an integer, or a string whose last
            whitespace-separated token is the residue index (e.g. ``"Res 42"``).

    Returns:
        int: The residue index.
    """
    return int(float(str(label).split()[-1]))


def canonical_pair(pair: Sequence[object]) -> Tuple[int, int]:
    """Normalise a residue pair into a sorted ``(res_low, res_high)`` integer tuple.

    Args:
        pair (Sequence[object]): Two residue identifiers, each accepted by
            :func:`residue_id`.

    Returns:
        Tuple[int, int]: The two residue indices as integers, sorted ascending.
    """
    ids = [residue_id(item) for item in pair]
    return tuple(sorted(ids))
