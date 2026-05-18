"""Round-trip tests for persistence helpers."""

from pathlib import Path

import numpy as np
import pytest

from grm_tcm_persistence import (
    canonicalize_eigvec_signs,
    compute_input_hashes,
    load_int_keyed_npz,
    load_joblib,
    manifest_sha,
    read_manifest,
    save_int_keyed_npz,
    save_joblib,
    write_manifest,
)


def test_manifest_round_trip(tmp_path: Path):
    write_manifest(
        tmp_path,
        config={"a": 1, "b": [1, 2]},
        inputs=[],
        schema_version="test-v1",
        random_seed=7,
    )
    m = read_manifest(tmp_path, allowed_schema_versions=["test-v1"])
    assert m["random_seed"] == 7
    assert m["schema_version"] == "test-v1"
    assert m["config"] == {"a": 1, "b": [1, 2]}
    assert "created_at" in m
    assert "git_commit" in m
    assert "package_versions" in m


def test_manifest_schema_version_check(tmp_path: Path):
    write_manifest(tmp_path, config={}, inputs=[], schema_version="bogus")
    with pytest.raises(ValueError, match="schema_version"):
        read_manifest(tmp_path, allowed_schema_versions=["expected-v1"])


def test_manifest_sha_is_stable(tmp_path: Path):
    write_manifest(tmp_path, config={"x": 1}, inputs=[], schema_version="test-v1")
    sha1 = manifest_sha(tmp_path)
    sha2 = manifest_sha(tmp_path)
    assert sha1 == sha2
    assert len(sha1) == 64


def test_int_keyed_npz_round_trip(tmp_path: Path):
    payload = {1: np.eye(3), 5: np.arange(4).astype(float), 42: np.array([[1.0, 2.0]])}
    save_int_keyed_npz(payload, tmp_path / "d.npz", prefix="k")
    loaded = load_int_keyed_npz(tmp_path / "d.npz", prefix="k")
    assert set(loaded) == set(payload)
    for k in payload:
        assert np.allclose(loaded[k], payload[k])


def test_joblib_round_trip(tmp_path: Path):
    obj = {"a": np.arange(5), "b": "hello"}
    save_joblib(obj, tmp_path / "o.joblib")
    back = load_joblib(tmp_path / "o.joblib")
    assert np.array_equal(back["a"], obj["a"])
    assert back["b"] == obj["b"]


def test_canonicalize_signs_flips_negative_leading_entry():
    M = np.array([[-1.0, 2.0], [3.0, 4.0]])
    F = canonicalize_eigvec_signs(M)
    assert F[0, 0] > 0
    assert F[0, 1] > 0
    assert np.allclose(F[:, 0], -M[:, 0])
    assert np.allclose(F[:, 1], M[:, 1])


def test_canonicalize_signs_skips_zero_column():
    M = np.zeros((3, 2))
    M[:, 0] = [-1.0, 2.0, 3.0]
    F = canonicalize_eigvec_signs(M)
    assert F[0, 0] > 0
    assert np.allclose(F[:, 1], 0.0)


def test_input_hashes_marks_absent(tmp_path: Path):
    real = tmp_path / "present.txt"
    real.write_text("hello")
    absent = tmp_path / "missing.txt"
    h = compute_input_hashes([real, absent])
    assert h[str(real)] != "absent"
    assert h[str(absent)] == "absent"
