"""Helpers for keeping generated artifact paths portable on Windows."""

from __future__ import annotations

import hashlib
from pathlib import Path


def bounded_artifact_path(
    directory: str | Path,
    base_name: str,
    suffix: str,
    *,
    max_absolute_chars: int = 240,
) -> str:
    """Return a deterministic artifact path with headroom below ``MAX_PATH``.

    The readable prefix is retained and a digest of the complete base name is
    appended whenever the desired absolute path would be too long.  Using a
    digest prevents different experiment names with the same prefix from
    overwriting one another.
    """
    directory_path = Path(directory)
    desired = directory_path / f"{base_name}{suffix}"
    if len(str(desired.resolve(strict=False))) <= max_absolute_chars:
        return str(desired)

    digest = hashlib.sha256(base_name.encode("utf-8")).hexdigest()[:12]
    absolute_directory = str(directory_path.resolve(strict=False))
    separator_chars = 1
    digest_separator_chars = 1
    prefix_budget = (
        max_absolute_chars
        - len(absolute_directory)
        - separator_chars
        - digest_separator_chars
        - len(digest)
        - len(suffix)
    )
    if prefix_budget < 1:
        raise ValueError(
            f"Output directory is too long for a bounded artifact path: {absolute_directory}"
        )

    readable_prefix = base_name[:prefix_budget].rstrip("._- ") or "artifact"
    compact = directory_path / f"{readable_prefix}_{digest}{suffix}"
    if len(str(compact.resolve(strict=False))) > max_absolute_chars:
        raise ValueError(f"Could not bound artifact path below {max_absolute_chars} characters")
    return str(compact)


__all__ = ["bounded_artifact_path"]
