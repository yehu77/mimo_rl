from __future__ import annotations


def test_package_import_does_not_require_gpu_or_engine() -> None:
    import mimo_rl

    assert mimo_rl.__version__ == "0.1.0"
