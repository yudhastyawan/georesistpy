"""
Format definitions and schema helpers for geoelectrical data files.

Each format is described as a :class:`FormatSpec` that documents the
expected columns, separator, and any special parsing rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FormatSpec:
    """Specification for a single data format.

    Attributes
    ----------
    name : str
        Human-readable name (e.g. ``"ABEM Terrameter"``).
    extension : str
        Typical file extension (e.g. ``".dat"``).
    separator : str | None
        Column separator; ``None`` means whitespace.
    required_columns : list of str
        Canonical column names that **must** be present.
    optional_columns : list of str
        Extra columns the format *may* contain.
    comment_char : str
        Character introducing comment lines.
    skip_rows : int
        Number of header rows to skip.
    notes : str
        Free-form description.
    """

    name: str
    extension: str = ".csv"
    separator: Optional[str] = ","
    required_columns: List[str] = field(default_factory=lambda: ["a", "b", "m", "n", "rhoa"])
    optional_columns: List[str] = field(default_factory=list)
    comment_char: str = "#"
    skip_rows: int = 0
    notes: str = ""


# ---------------------------------------------------------------------------
# Built-in format catalog
# ---------------------------------------------------------------------------

FORMAT_CATALOG: Dict[str, FormatSpec] = {
    "csv": FormatSpec(
        name="Generic CSV",
        extension=".csv",
        separator=",",
        notes="Comma-separated with a header row.",
    ),
    "txt": FormatSpec(
        name="Generic TXT",
        extension=".txt",
        separator=None,
        notes="Whitespace-separated with optional '#' comments.",
    ),
    "abem": FormatSpec(
        name="ABEM Terrameter",
        extension=".dat",
        separator=None,
        optional_columns=["i", "v", "error"],
        notes="ABEM SAS/LS export format with header block.",
    ),
    "syscal": FormatSpec(
        name="Iris Syscal Pro",
        extension=".txt",
        separator=None,
        optional_columns=["i", "v", "error", "sp"],
        notes="Syscal Pro/R8 export; metadata lines separated by dashes.",
    ),
}
