#!/usr/bin/env python3
r"""
download_venue_template.py
Fetches official conference/journal LaTeX templates on demand and stages
them into a paper's working directory (typically `<paper>/final/`).

Architecture
============
- VENUE_REGISTRY maps a venue alias -> {source URL, archive type, files to
  extract, recommended \documentclass / \usepackage / bibstyle} for that
  venue.
- The script downloads each archive at most once and caches it under
  ~/.cache/venue-templates/<venue>/. Subsequent calls reuse the cache.
- Files matching the registry's `files_to_extract` glob list are copied into
  the requested output directory.
- A JSON "manifest" is printed to stdout so callers (the writer Agent and
  pipeline) can pick the right \documentclass line and bib style without
  re-encoding venue knowledge in their prompts.

Usage
=====
    # Fetch ACL/EMNLP/NAACL style files into a paper's final/ directory
    python download_venue_template.py --venue acl --output-dir /path/to/final

    # ICLR template for a specific year
    python download_venue_template.py --venue iclr --year 2025 --output-dir /path/to/final

    # Inspect supported venues without downloading
    python download_venue_template.py --list

    # Print the manifest (no copy) — useful for picking documentclass
    python download_venue_template.py --venue acl --print-manifest

Manifest schema (JSON on stdout when --output-dir is provided)
==============================================================
{
  "venue": "acl",
  "year": null,
  "documentclass": "\\documentclass[11pt]{article}",
  "usepackages": ["\\usepackage[final]{acl}"],
  "bibstyle": "acl_natbib",
  "extracted_files": ["acl.sty", "acl_natbib.bst"],
  "output_dir": "/path/to/final",
  "source": "https://github.com/acl-org/acl-style-files/...",
  "cache_dir": "/home/.../.cache/venue-templates/acl"
}

Adding a new venue
==================
Append an entry to VENUE_REGISTRY. Required keys:
- aliases:           list of names that map to this entry (e.g. ["acl","emnlp"])
- source_url:        URL of an archive (.zip / .tar.gz)
- archive_type:      "zip" or "tar.gz"
- files_to_extract:  list of glob patterns relative to the extracted archive
                     (e.g. "*.sty", "acl_natbib.bst", "lncs/llncs.cls")
- documentclass:     LaTeX \documentclass line the writer should use
- usepackages:       list of \usepackage lines (may be empty)
- bibstyle:          BibTeX style name (without .bst), or null
- description:       short human-readable blurb
- page_limit:        free-text page limit
- anonymization:     "Required (double-blind)" / "None" / etc.

For year-dependent venues (ICLR, NeurIPS, AAAI...) parameterise source_url
with `{year}`; the script substitutes --year (default = registry default).
"""

from __future__ import annotations

import argparse
import fnmatch
import io
import json
import os
import re
import shutil
import sys
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


# -------------------------------------------------------------------------
# Registry
# -------------------------------------------------------------------------
# Each entry describes how to obtain and stage one venue's official template.
# URLs marked "verified" have been used at least once successfully. Others
# are documented best-effort sources; if a download fails the script prints
# clear instructions for manual fallback (drop files into the cache dir).

VENUE_REGISTRY: dict = {
    # ----- ACL family (ACL/EMNLP/NAACL/EACL/Findings) ----------------------
    # All four conferences share the acl-org/acl-style-files repo. Verified
    # working as of 2026-04 (used by the 20260425_150043 paper run).
    "acl": {
        "aliases": ["acl", "emnlp", "naacl", "eacl", "findings"],
        "source_url": "https://github.com/acl-org/acl-style-files/archive/refs/heads/master.zip",
        "archive_type": "zip",
        "files_to_extract": ["*acl-style-files-master/acl.sty",
                             "*acl-style-files-master/acl_natbib.bst"],
        "rename_map": {"acl-style-files-master/acl.sty": "acl.sty",
                       "acl-style-files-master/acl_natbib.bst": "acl_natbib.bst"},
        "documentclass": r"\documentclass[11pt]{article}",
        "usepackages": [r"\usepackage[final]{acl}",
                        r"\usepackage{times}",
                        r"\usepackage{latexsym}"],
        "bibstyle": "acl_natbib",
        "description": "ACL/EMNLP/NAACL/EACL conferences (Computational Linguistics)",
        "page_limit": "8 pages long / 4 pages short, refs unlimited",
        "anonymization": "Required (double-blind during review)",
        "default_year": None,
        "verified": True,
    },

    # ----- ICLR -----------------------------------------------------------
    # ICLR templates are published per-year on the ICLR site / GitHub. The
    # repo at openreview-py/iclr_template is most reliable; we fall back to
    # the official year-specific zip on iclr.cc when set.
    "iclr": {
        "aliases": ["iclr"],
        "source_url": "https://media.iclr.cc/Conferences/ICLR{year}/iclr{year}.zip",
        "archive_type": "zip",
        "files_to_extract": ["*iclr{year}_conference.sty",
                             "*iclr{year}_conference.bst",
                             "*natbib.sty",
                             "*fancyhdr.sty"],
        "documentclass_template": r"\documentclass{{article}}",
        "usepackages_template": [r"\usepackage{{iclr{year}_conference}}"],
        "bibstyle_template": "iclr{year}_conference",
        "description": "International Conference on Learning Representations",
        "page_limit": "9 pages main + unlimited refs/appendix",
        "anonymization": "Required (double-blind)",
        "default_year": 2025,
        "verified": False,
    },

    # ----- NeurIPS --------------------------------------------------------
    "neurips": {
        "aliases": ["neurips", "nips"],
        "source_url": "https://media.neurips.cc/Conferences/NeurIPS{year}/Styles.zip",
        "archive_type": "zip",
        "files_to_extract": ["*neurips_{year}.sty",
                             "*neurips_{year}.tex"],
        "documentclass_template": r"\documentclass{{article}}",
        "usepackages_template": [r"\usepackage[final]{{neurips_{year}}}"],
        "bibstyle_template": "plainnat",
        "description": "Neural Information Processing Systems",
        "page_limit": "9 pages main + unlimited refs/appendix",
        "anonymization": "Required (double-blind)",
        "default_year": 2024,
        "verified": False,
    },

    # ----- ICML -----------------------------------------------------------
    "icml": {
        "aliases": ["icml"],
        "source_url": "https://media.icml.cc/Conferences/ICML{year}/Styles/icml{year}.zip",
        "archive_type": "zip",
        "files_to_extract": ["*icml{year}.sty",
                             "*icml{year}.bst",
                             "*fancyhdr.sty"],
        "documentclass_template": r"\documentclass{{article}}",
        "usepackages_template": [r"\usepackage{{icml{year}}}"],
        "bibstyle_template": "icml{year}",
        "description": "International Conference on Machine Learning",
        "page_limit": "8 pages main + unlimited refs",
        "anonymization": "Required (double-blind)",
        "default_year": 2024,
        "verified": False,
    },

    # ----- AAAI -----------------------------------------------------------
    "aaai": {
        "aliases": ["aaai"],
        "source_url": "https://aaai.org/wp-content/uploads/{year}/01/AuthorKit{year_short}.zip",
        "archive_type": "zip",
        "files_to_extract": ["*aaai{year_short}.sty",
                             "*aaai{year_short}.bst"],
        "documentclass_template": r"\documentclass[letterpaper]{{article}}",
        "usepackages_template": [r"\usepackage{{aaai{year_short}}}"],
        "bibstyle_template": "aaai{year_short}",
        "description": "Association for the Advancement of Artificial Intelligence",
        "page_limit": "7 pages main + 2 pages refs",
        "anonymization": "Required (double-blind)",
        "default_year": 2025,
        "verified": False,
    },

    # ----- IJCAI ----------------------------------------------------------
    "ijcai": {
        "aliases": ["ijcai"],
        "source_url": "https://www.ijcai.org/proceedings/{year_short}/IJCAI-{year_short}-Author-Kit.zip",
        "archive_type": "zip",
        "files_to_extract": ["*ijcai{year_short}.sty",
                             "*named.bst",
                             "*ijcai{year_short}.bst"],
        "documentclass_template": r"\documentclass{{article}}",
        "usepackages_template": [r"\usepackage{{ijcai{year_short}}}"],
        "bibstyle_template": "named",
        "description": "International Joint Conference on Artificial Intelligence",
        "page_limit": "7 pages main + 2 pages refs",
        "anonymization": "Required (double-blind)",
        "default_year": 2024,
        "verified": False,
    },

    # ----- CVPR / ICCV (CVF conferences) ----------------------------------
    "cvpr": {
        "aliases": ["cvpr", "iccv", "wacv", "cvf"],
        "source_url": "https://github.com/cvpr-org/author-kit/archive/refs/heads/main.zip",
        "archive_type": "zip",
        "files_to_extract": ["*author-kit-main/*.sty",
                             "*author-kit-main/*.cls",
                             "*author-kit-main/*.bst"],
        "documentclass": r"\documentclass[10pt,twocolumn,letterpaper]{article}",
        "usepackages": [r"\usepackage{cvpr}"],
        "bibstyle": "ieee_fullname",
        "description": "IEEE/CVF Computer Vision conferences",
        "page_limit": "8 pages main + unlimited refs",
        "anonymization": "Required (double-blind)",
        "default_year": None,
        "verified": False,
    },

    # ----- IEEE (general IEEEtran) ----------------------------------------
    # IEEEtran lives on CTAN under a stable URL. Use this for any IEEE
    # transactions, magazines, conferences (ICASSP, ICRA, ICC, etc).
    "ieee": {
        "aliases": ["ieee", "ieeetran", "icassp", "icra", "icc", "globecom"],
        "source_url": "https://mirrors.ctan.org/macros/latex/contrib/IEEEtran.zip",
        "archive_type": "zip",
        "files_to_extract": ["*IEEEtran/IEEEtran.cls",
                             "*IEEEtran/IEEEtran.bst",
                             "*IEEEtran/bibtex/IEEEtran.bst"],
        "documentclass": r"\documentclass[conference]{IEEEtran}",
        "usepackages": [r"\usepackage{cite}",
                        r"\usepackage{amsmath,amssymb,amsfonts}",
                        r"\usepackage{algorithmic}",
                        r"\usepackage{graphicx}"],
        "bibstyle": "IEEEtran",
        "description": "IEEE conferences and journals (general IEEEtran class)",
        "page_limit": "Varies by venue (typ. 6-10 pages)",
        "anonymization": "Optional (depends on venue)",
        "default_year": None,
        "verified": True,  # CTAN mirror is stable
    },

    # ----- ACM (acmart class) ---------------------------------------------
    # acmart on CTAN.
    "acm": {
        "aliases": ["acm", "acmart", "sigchi", "chi", "kdd", "sigir", "www", "uist"],
        "source_url": "https://mirrors.ctan.org/macros/latex/contrib/acmart.zip",
        "archive_type": "zip",
        "files_to_extract": ["*acmart/acmart.cls",
                             "*acmart/ACM-Reference-Format.bst",
                             "*acmart/acmart.dtx"],
        "documentclass": r"\documentclass[sigconf,review,anonymous]{acmart}",
        "usepackages": [],
        "bibstyle": "ACM-Reference-Format",
        "description": "ACM conferences and journals (acmart class)",
        "page_limit": "Varies by venue",
        "anonymization": "Required for review (double-blind)",
        "default_year": None,
        "verified": True,
    },

    # ----- Springer LNCS --------------------------------------------------
    "lncs": {
        "aliases": ["lncs", "springer", "miccai", "eccv"],
        "source_url": "https://mirrors.ctan.org/macros/latex/contrib/llncs.zip",
        "archive_type": "zip",
        "files_to_extract": ["*llncs/llncs.cls",
                             "*llncs/splncs04.bst"],
        "documentclass": r"\documentclass[runningheads]{llncs}",
        "usepackages": [r"\usepackage{graphicx}"],
        "bibstyle": "splncs04",
        "description": "Springer Lecture Notes in Computer Science (MICCAI, ECCV, etc.)",
        "page_limit": "Typ. 12-15 pages incl. refs",
        "anonymization": "Optional",
        "default_year": None,
        "verified": True,
    },
}


# -------------------------------------------------------------------------
# Core
# -------------------------------------------------------------------------

@dataclass
class VenueSpec:
    """Resolved venue specification with year substitution applied."""
    canonical: str
    aliases: list
    source_url: str
    archive_type: str
    files_to_extract: list
    rename_map: dict
    documentclass: str
    usepackages: list
    bibstyle: Optional[str]
    description: str
    page_limit: str
    anonymization: str
    year: Optional[int]
    verified: bool


def _resolve_venue(name: str, year: Optional[int] = None) -> VenueSpec:
    """Look up `name` in VENUE_REGISTRY and substitute year placeholders."""
    name_lc = name.lower().strip()
    canonical = None
    entry = None
    for key, val in VENUE_REGISTRY.items():
        if name_lc == key or name_lc in [a.lower() for a in val["aliases"]]:
            canonical = key
            entry = val
            break
    if entry is None:
        raise ValueError(
            f"Unknown venue '{name}'. Run with --list to see supported venues."
        )

    use_year = year if year is not None else entry.get("default_year")
    fmt = {"year": use_year, "year_short": str(use_year)[-2:] if use_year else ""}

    def sub(s: str) -> str:
        if s is None:
            return None
        try:
            return s.format(**fmt)
        except (KeyError, IndexError):
            return s

    documentclass = entry.get("documentclass") or sub(entry.get("documentclass_template", ""))
    usepackages_src = entry.get("usepackages")
    if usepackages_src is None:
        usepackages_src = [sub(u) for u in entry.get("usepackages_template", [])]
    bibstyle = entry.get("bibstyle")
    if bibstyle is None and "bibstyle_template" in entry:
        bibstyle = sub(entry["bibstyle_template"])

    return VenueSpec(
        canonical=canonical,
        aliases=entry["aliases"],
        source_url=sub(entry["source_url"]),
        archive_type=entry["archive_type"],
        files_to_extract=[sub(p) for p in entry["files_to_extract"]],
        rename_map={sub(k): sub(v) for k, v in entry.get("rename_map", {}).items()},
        documentclass=documentclass,
        usepackages=usepackages_src,
        bibstyle=bibstyle,
        description=entry["description"],
        page_limit=entry["page_limit"],
        anonymization=entry["anonymization"],
        year=use_year,
        verified=entry.get("verified", False),
    )


def _cache_root() -> Path:
    """~/.cache/venue-templates/ (override via $VENUE_TEMPLATES_CACHE)."""
    custom = os.environ.get("VENUE_TEMPLATES_CACHE")
    if custom:
        return Path(custom).expanduser().resolve()
    return Path.home() / ".cache" / "venue-templates"


def _venue_cache_dir(spec: VenueSpec) -> Path:
    suffix = f"_{spec.year}" if spec.year else ""
    return _cache_root() / f"{spec.canonical}{suffix}"


def _download_archive(spec: VenueSpec, force: bool = False) -> Path:
    """Download spec.source_url into the venue cache, return archive path.

    If `force=False` and the archive already exists, returns the cached copy.
    """
    cache_dir = _venue_cache_dir(spec)
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = ".zip" if spec.archive_type == "zip" else ".tar.gz"
    archive_path = cache_dir / f"archive{ext}"

    if archive_path.exists() and not force:
        return archive_path

    print(f"[download] {spec.canonical}: GET {spec.source_url}", file=sys.stderr)
    req = Request(spec.source_url, headers={"User-Agent": "venue-templates/1.0"})
    try:
        with urlopen(req, timeout=60) as resp:
            data = resp.read()
    except (HTTPError, URLError, TimeoutError) as e:
        manual = cache_dir / "manual"
        raise RuntimeError(
            f"Failed to download '{spec.canonical}' template from {spec.source_url}: {e}.\n"
            f"Manual fallback: drop the venue's .sty/.cls/.bst files into\n"
            f"  {manual}/\n"
            f"and rerun this command — files in manual/ are used when download fails."
        ) from e

    archive_path.write_bytes(data)
    return archive_path


def _open_archive(archive_path: Path, archive_type: str):
    """Yield (member_name, file_obj) for each file in the archive."""
    if archive_type == "zip":
        with zipfile.ZipFile(archive_path) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                yield info.filename, z.read(info.filename)
    else:  # tar.gz
        with tarfile.open(archive_path, "r:gz") as t:
            for member in t.getmembers():
                if not member.isfile():
                    continue
                f = t.extractfile(member)
                if f is None:
                    continue
                yield member.name, f.read()


def _match_any(name: str, patterns: list) -> bool:
    return any(fnmatch.fnmatch(name, p) for p in patterns)


def _extract_files(spec: VenueSpec, archive_path: Path, output_dir: Path) -> list:
    """Copy files matching spec.files_to_extract from the archive into output_dir.

    Returns the list of (relative) output filenames written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    seen = set()

    for member_name, blob in _open_archive(archive_path, spec.archive_type):
        if not _match_any(member_name, spec.files_to_extract):
            continue
        # Decide output filename: rename_map honoured first, else basename
        target_name = None
        for src_pat, dst in spec.rename_map.items():
            if fnmatch.fnmatch(member_name, src_pat) or member_name.endswith(src_pat):
                target_name = dst
                break
        if target_name is None:
            target_name = Path(member_name).name
        if target_name in seen:
            continue  # dedup if multiple archive members map to same name
        seen.add(target_name)

        out_path = output_dir / target_name
        out_path.write_bytes(blob)
        written.append(target_name)

    # Apply manual-fallback overlay if user dropped files into cache/manual/
    manual_dir = _venue_cache_dir(spec) / "manual"
    if manual_dir.is_dir():
        for f in manual_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, output_dir / f.name)
                if f.name not in written:
                    written.append(f.name)
    return written


def fetch_venue_template(
    venue: str,
    output_dir: Path,
    year: Optional[int] = None,
    force_download: bool = False,
) -> dict:
    """Download (if needed) and stage `venue` template into `output_dir`.

    Returns a manifest dict (also see module docstring).
    """
    spec = _resolve_venue(venue, year=year)
    archive_path = _download_archive(spec, force=force_download)
    written = _extract_files(spec, archive_path, output_dir)

    if not written:
        manual = _venue_cache_dir(spec) / "manual"
        raise RuntimeError(
            f"No files matching {spec.files_to_extract} were found in the\n"
            f"downloaded archive for '{spec.canonical}'. The venue's template\n"
            f"layout may have changed. Either:\n"
            f"  1. Update files_to_extract in download_venue_template.py, or\n"
            f"  2. Drop the .sty/.cls/.bst files into {manual}/ and rerun.\n"
            f"Archive cached at: {archive_path}"
        )

    return {
        "venue": spec.canonical,
        "year": spec.year,
        "documentclass": spec.documentclass,
        "usepackages": spec.usepackages,
        "bibstyle": spec.bibstyle,
        "extracted_files": written,
        "output_dir": str(output_dir.resolve()),
        "source": spec.source_url,
        "cache_dir": str(_venue_cache_dir(spec)),
        "page_limit": spec.page_limit,
        "anonymization": spec.anonymization,
        "verified": spec.verified,
    }


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------

def _list_venues() -> None:
    print("Supported venues (use any alias as --venue):\n")
    for canonical, entry in VENUE_REGISTRY.items():
        aliases = ", ".join(entry["aliases"])
        verified = "✓" if entry.get("verified") else " "
        year_note = (f"  default year: {entry['default_year']}"
                     if entry.get("default_year") else "")
        print(f"  [{verified}] {canonical:8s}  aliases: {aliases}")
        print(f"        {entry['description']}")
        print(f"        page limit: {entry['page_limit']}")
        print(f"        anonymisation: {entry['anonymization']}{year_note}")
        print()
    print("Legend: ✓ = source URL verified working; blank = best-effort URL,")
    print("manual fallback may be required if the venue updates its template.")


def main() -> int:
    p = argparse.ArgumentParser(
        description=("Download and stage official conference/journal LaTeX "
                     "templates into a paper's output directory."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  download_venue_template.py --venue acl --output-dir ./final\n"
            "  download_venue_template.py --venue iclr --year 2025 "
            "--output-dir ./final\n"
            "  download_venue_template.py --list\n"
        ),
    )
    p.add_argument("--venue", help="Venue alias (acl, iclr, neurips, cvpr, ieee, acm, lncs, ...)")
    p.add_argument("--output-dir", help="Where to copy the extracted .sty/.cls/.bst files")
    p.add_argument("--year", type=int, help="Override default year (for year-versioned venues)")
    p.add_argument("--force", action="store_true", help="Re-download even if cached")
    p.add_argument("--list", action="store_true", help="List supported venues and exit")
    p.add_argument("--print-manifest", action="store_true",
                   help="Print manifest JSON without copying files")
    args = p.parse_args()

    if args.list:
        _list_venues()
        return 0

    if not args.venue:
        p.error("--venue is required (or use --list)")

    if args.print_manifest:
        spec = _resolve_venue(args.venue, year=args.year)
        manifest = {
            "venue": spec.canonical,
            "year": spec.year,
            "documentclass": spec.documentclass,
            "usepackages": spec.usepackages,
            "bibstyle": spec.bibstyle,
            "source": spec.source_url,
            "verified": spec.verified,
        }
        print(json.dumps(manifest, indent=2))
        return 0

    if not args.output_dir:
        p.error("--output-dir is required (or use --print-manifest)")

    output_dir = Path(args.output_dir).expanduser().resolve()
    try:
        manifest = fetch_venue_template(
            args.venue, output_dir,
            year=args.year, force_download=args.force,
        )
    except (ValueError, RuntimeError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
