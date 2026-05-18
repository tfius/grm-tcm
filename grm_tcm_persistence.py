from __future__ import annotations

"""
Shared persistence helpers for the GRM-TCM static and dynamic pipelines.

Scope:
- manifest.json assembly (config + git + input hashes + package versions)
- joblib + npz convenience wrappers
- keyed-by-int npz helpers (since npz keys must be strings)

These helpers are deliberately dependency-light: only stdlib + numpy + joblib.
"""

import hashlib
import json
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import joblib
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guard
    raise ModuleNotFoundError(
        "joblib is required for GRM-TCM persistence. Install via `uv sync`."
    ) from exc


MANIFEST_FILENAME = "manifest.json"


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_input_hashes(paths: Iterable[Path]) -> Dict[str, str]:
    """Hash each existing input. Missing files map to 'absent'."""

    out: Dict[str, str] = {}
    for p in paths:
        p = Path(p)
        out[str(p)] = sha256_file(p) if p.exists() else "absent"
    return out


def git_commit_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Best-effort git provenance. Returns {'sha': None, 'dirty': None} outside a repo."""

    repo_root = Path(repo_root) if repo_root else Path.cwd()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"sha": None, "dirty": None}
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo_root, stderr=subprocess.DEVNULL, text=True
        )
        dirty = bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        dirty = None
    return {"sha": sha, "dirty": dirty}


def package_versions(names: Iterable[str]) -> Dict[str, str]:
    """Return installed versions for the requested packages."""

    out: Dict[str, str] = {}
    for name in names:
        try:
            out[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            out[name] = "absent"
    return out


def _config_to_dict(config: Any) -> Dict[str, Any]:
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    raise TypeError(f"config must be a dataclass or dict, got {type(config)!r}")


def write_manifest(
    model_dir: Path,
    *,
    config: Any,
    inputs: Iterable[Path],
    schema_version: str,
    random_seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write manifest.json into model_dir and return the manifest dict."""

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "schema_version": schema_version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "random_seed": random_seed,
        "config": _config_to_dict(config),
        "git_commit": git_commit_info(),
        "input_hashes": compute_input_hashes(inputs),
        "package_versions": package_versions(["numpy", "scipy", "scikit-learn", "pandas", "joblib"]),
    }
    if extra:
        manifest["extra"] = extra
    path = model_dir / MANIFEST_FILENAME
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def read_manifest(model_dir: Path, *, allowed_schema_versions: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """Read manifest.json and optionally assert schema_version is in allowed set."""

    path = Path(model_dir) / MANIFEST_FILENAME
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    if allowed_schema_versions is not None:
        allowed: List[str] = list(allowed_schema_versions)
        if manifest.get("schema_version") not in allowed:
            raise ValueError(
                f"Manifest schema_version {manifest.get('schema_version')!r} "
                f"not in supported set {allowed}"
            )
    return manifest


def manifest_sha(model_dir: Path) -> str:
    """Hash the canonical manifest.json bytes so cross-references stay stable."""

    return sha256_file(Path(model_dir) / MANIFEST_FILENAME)


def save_joblib(obj: Any, path: Path) -> None:
    """Persist an arbitrary Python object via joblib."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_joblib(path: Path) -> Any:
    """Load a joblib-serialized object."""

    return joblib.load(path)


def save_int_keyed_npz(d: Dict[int, np.ndarray], path: Path, prefix: str = "k") -> None:
    """Save a dict[int, ndarray] as a single .npz, encoding int keys as f'{prefix}{int}'."""

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{f"{prefix}{int(k)}": np.asarray(v) for k, v in d.items()})


def load_int_keyed_npz(path: Path, prefix: str = "k") -> Dict[int, np.ndarray]:
    """Inverse of save_int_keyed_npz."""

    with np.load(path) as f:
        out: Dict[int, np.ndarray] = {}
        for name in f.files:
            if not name.startswith(prefix):
                continue
            key = int(name[len(prefix):])
            out[key] = f[name].copy()
    return out


def canonicalize_eigvec_signs(eigenvectors: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Flip each column so its first |entry|>tol element is positive.

    Why: eigsh/eigh do not pin eigenvector signs. Without canonicalization, repeat
    decompositions of the same Laplacian may flip mode_k -> -mode_k, which silently
    inverts any per-mode scatter or correlation. The GRM resonance matrix G itself
    is sign-invariant (quadratic in psi), so this flip is purely cosmetic for G but
    important for grm_mode_* CSV stability.
    """

    out = eigenvectors.copy()
    for j in range(out.shape[1]):
        col = out[:, j]
        idx = np.argmax(np.abs(col) > tol)
        if np.abs(col[idx]) <= tol:
            continue
        if col[idx] < 0:
            out[:, j] = -col
    return out
