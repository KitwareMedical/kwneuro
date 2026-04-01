from __future__ import annotations

from pathlib import Path

import pytest

from kwneuro.cache import (
    CacheSpec,
    PipelineCache,
    _active_cache,
    cacheable,
)


class FakeResource:
    """Minimal in-memory resource with a cache protocol."""

    def __init__(self, value: int) -> None:
        self.value = value

    @classmethod
    def _cache_files(cls, step_name: str) -> list[str]:
        return [f"{step_name}.fake"]

    def _cache_save(self, cache_dir: Path, step_name: str) -> None:
        (cache_dir / f"{step_name}.fake").write_text(str(self.value))

    @classmethod
    def _cache_load(cls, cache_dir: Path, step_name: str) -> FakeResource:
        val = int((cache_dir / f"{step_name}.fake").read_text())
        return cls(val)


class NoCacheProtocol:
    """Stub without a cache protocol — should trigger TypeError."""


# Exmaple cacheable function
@cacheable
def compute_fake(x: int, multiplier: int = 2) -> FakeResource:
    return FakeResource(x * multiplier)


class TestIsForced:
    def test_false(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path, force=False)
        assert not pc.is_forced("any_step")

    def test_true(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path, force=True)
        assert pc.is_forced("any_step")

    def test_set_match(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path, force={"step_a"})
        assert pc.is_forced("step_a")

    def test_set_no_match(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path, force={"step_b"})
        assert not pc.is_forced("step_a")


# ---------------------------------------------------------------------------
# PipelineCache.is_cached
# ---------------------------------------------------------------------------


class TestIsCached:
    def test_missing_file(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path)
        assert not pc.is_cached("step", ["missing.txt"])

    def test_all_files_present(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("x")
        pc = PipelineCache(tmp_path, force=False)
        assert pc.is_cached("step", ["out.txt"])

    def test_forced_overrides_present(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("x")
        pc = PipelineCache(tmp_path, force=True)
        assert not pc.is_cached("step", ["out.txt"])

    def test_params_match(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("x")
        import json

        (tmp_path / "step.params.json").write_text(json.dumps({"x": 1}, sort_keys=True))
        pc = PipelineCache(tmp_path)
        assert pc.is_cached("step", ["out.txt"], params={"x": 1})

    def test_params_mismatch(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("x")
        import json

        (tmp_path / "step.params.json").write_text(json.dumps({"x": 1}, sort_keys=True))
        pc = PipelineCache(tmp_path)
        assert not pc.is_cached("step", ["out.txt"], params={"x": 99})

    def test_params_file_missing_is_miss(self, tmp_path: Path) -> None:
        (tmp_path / "out.txt").write_text("x")
        pc = PipelineCache(tmp_path)
        assert not pc.is_cached("step", ["out.txt"], params={"x": 1})


# ---------------------------------------------------------------------------
# Context manager
# ---------------------------------------------------------------------------


class TestContextManager:
    def test_sets_active_cache(self, tmp_path: Path) -> None:
        assert _active_cache.get() is None
        pc = PipelineCache(tmp_path)
        with pc:
            assert _active_cache.get() is pc
        assert _active_cache.get() is None

    def test_restores_on_exception(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path)
        try:
            with pc:
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        assert _active_cache.get() is None

    def test_creates_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "nested" / "cache"
        assert not cache_dir.exists()
        with PipelineCache(cache_dir):
            assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# @cacheable (bare, auto-spec)
# ---------------------------------------------------------------------------


class TestCacheable:
    def test_cache_miss_calls_function(self, tmp_path: Path) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int) -> FakeResource:
            calls.append(x)
            return FakeResource(x * 3)

        with PipelineCache(tmp_path):
            result = fn(4)

        assert calls == [4]
        assert result.value == 12

    def test_cache_hit_skips_function(self, tmp_path: Path) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int) -> FakeResource:
            calls.append(x)
            return FakeResource(x * 3)

        pc = PipelineCache(tmp_path)
        with pc:
            fn(4)  # populate cache
        calls.clear()
        with pc:
            result = fn(99)  # different arg — cache hit returns saved value

        assert calls == []  # function not called
        assert result.value == 12  # loaded from cache (original value)

    def test_outside_context_no_caching(self) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int) -> FakeResource:
            calls.append(x)
            return FakeResource(x)

        result = fn(7)
        assert calls == [7]
        assert result.value == 7  # returned directly, not from disk

    def test_parameter_change_triggers_recompute(self, tmp_path: Path) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int, multiplier: int = 2) -> FakeResource:
            calls.append(multiplier)
            return FakeResource(x * multiplier)

        pc = PipelineCache(tmp_path)
        with pc:
            fn(5, multiplier=2)
        calls.clear()
        with pc:
            result = fn(5, multiplier=10)  # changed scalar param → recompute

        assert calls == [10]
        assert result.value == 50

    def test_same_params_no_recompute(self, tmp_path: Path) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int, multiplier: int = 2) -> FakeResource:
            calls.append(multiplier)
            return FakeResource(x * multiplier)

        pc = PipelineCache(tmp_path)
        with pc:
            fn(5, multiplier=2)
        calls.clear()
        with pc:
            fn(5, multiplier=2)  # same params → cache hit

        assert calls == []

    def test_type_error_on_no_return_annotation(self) -> None:
        with pytest.raises(TypeError, match="no resolvable return type"):

            @cacheable
            def bad_fn(x: int):
                return FakeResource(x)

    def test_type_error_on_missing_protocol(self) -> None:
        with pytest.raises(TypeError, match="does not implement the cache protocol"):

            @cacheable
            def bad_fn(x: int) -> NoCacheProtocol:
                return NoCacheProtocol()

    def test_functools_wraps_preserved(self) -> None:
        assert compute_fake.__name__ == "compute_fake"

    def test_force_step_bypasses_cache(self, tmp_path: Path) -> None:
        calls: list[int] = []

        @cacheable
        def fn(x: int) -> FakeResource:
            calls.append(x)
            return FakeResource(x)

        pc = PipelineCache(tmp_path)
        with pc:
            fn(1)
        calls.clear()
        with PipelineCache(tmp_path, force={"fn"}):
            fn(1)

        assert calls == [1]  # forced recompute


# ---------------------------------------------------------------------------
# @cacheable(CacheSpec(...)) — explicit spec
# ---------------------------------------------------------------------------
class TestCacheSpec:
    def test_explicit_spec_miss_and_hit(self, tmp_path: Path) -> None:
        calls: list[str] = []

        @cacheable(
            CacheSpec(
                files=["result.txt"],
                save=lambda val, d: (d / "result.txt").write_text(val),
                load=lambda d: (d / "result.txt").read_text(),
            )
        )
        def fn(label: str) -> str:
            calls.append(label)
            return label.upper()

        pc = PipelineCache(tmp_path)
        with pc:
            r1 = fn("hello")
        assert r1 == "HELLO"
        assert calls == ["hello"]

        calls.clear()
        with pc:
            r2 = fn("world")  # cache hit — returns previously saved value
        assert r2 == "HELLO"
        assert calls == []

    def test_explicit_spec_param_change(self, tmp_path: Path) -> None:
        calls: list[str] = []

        @cacheable(
            CacheSpec(
                files=["result.txt"],
                save=lambda val, d: (d / "result.txt").write_text(val),
                load=lambda d: (d / "result.txt").read_text(),
            )
        )
        def fn(label: str) -> str:
            calls.append(label)
            return label.upper()

        pc = PipelineCache(tmp_path)
        with pc:
            fn("hello")
        calls.clear()
        with pc:
            r = fn("goodbye")  # scalar param changed → recompute

        assert calls == ["goodbye"]
        assert r == "GOODBYE"


# ---------------------------------------------------------------------------
# PipelineCache.status
# ---------------------------------------------------------------------------
class TestStatus:
    def test_missing_shows_false(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path)
        result = pc.status([compute_fake])
        assert result["compute_fake"] is False

    def test_present_shows_true(self, tmp_path: Path) -> None:
        pc = PipelineCache(tmp_path)
        with pc:
            compute_fake(3)
        result = pc.status([compute_fake])
        assert result["compute_fake"] is True

    def test_non_cacheable_skipped(self, tmp_path: Path) -> None:
        def plain() -> None:
            pass

        pc = PipelineCache(tmp_path)
        assert pc.status([plain]) == {}

    def test_multiple_steps(self, tmp_path: Path) -> None:
        @cacheable
        def step_a(x: int) -> FakeResource:
            return FakeResource(x)

        @cacheable
        def step_b(x: int) -> FakeResource:
            return FakeResource(x)

        pc = PipelineCache(tmp_path)
        with pc:
            step_a(1)
        result = pc.status([step_a, step_b])
        assert result["step_a"] is True
        assert result["step_b"] is False
