from __future__ import annotations

import contextvars
import functools
import inspect
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

T = TypeVar("T")

_active_cache: contextvars.ContextVar[Cache | None] = contextvars.ContextVar(
    "_active_cache", default=None
)


@dataclass
class CacheSpec(Generic[T]):
    """Explicit cache spec for ``@cacheable`` when the return type cannot carry
    the cache protocol (e.g. tuples, plain arrays).

    Use the bare ``@cacheable`` decorator instead when the return type implements
    ``_cache_files`` / ``_cache_save`` / ``_cache_load``.
    """

    files: list[str]
    """Filenames relative to ``cache_dir`` that constitute the cached output."""

    save: Callable[[T, Path], None]
    """Persist *result* to *cache_dir*."""

    load: Callable[[Path], T]
    """Reconstruct the result from *cache_dir*."""

    step_name: str = ""
    """Step name override. Defaults to the decorated function's ``__name__``."""


@dataclass
class _CacheInfo:
    """Internal metadata attached to each @cacheable-decorated function."""

    step_name: str
    get_files: Callable[[str], list[str]]


@dataclass
class Cache:
    """Context manager that activates caching for all ``@cacheable``-decorated
    functions within a ``with`` block.

    ``cache_dir`` is created automatically. ``force`` controls cache bypass:
    ``False`` (default) uses all cached outputs; ``True`` recomputes everything;
    a ``set[str]`` of step names recomputes only those steps.

    Scalar arguments (``int``, ``float``, ``str``, ``bool``, ``None``) are
    fingerprinted — if they change between runs the step recomputes automatically.
    Non-scalar arguments are not tracked; use ``force={"step_name"}`` when those
    change.  Each subject should use a distinct ``cache_dir`` to avoid races.
    """

    cache_dir: Path
    force: bool | set[str] = field(default=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Cache:
        self._token = _active_cache.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        _active_cache.reset(self._token)

    def is_forced(self, step_name: str) -> bool:
        """Return ``True`` if this step should bypass the cache."""
        if isinstance(self.force, set):
            return step_name in self.force
        return bool(self.force)

    def is_cached(
        self,
        step_name: str,
        files: list[str],
        params: dict[str, Any] | None = None,
    ) -> bool:
        """Return ``True`` when all output files exist, the step is not forced,
        and (if *params* is given) the stored parameter fingerprint matches.
        """
        if self.is_forced(step_name):
            return False
        if not all((self.cache_dir / f).exists() for f in files):
            return False
        if params:
            params_file = self.cache_dir / f"{step_name}.params.json"
            if not params_file.exists():
                return False
            try:
                stored = json.loads(params_file.read_text(encoding="utf-8"))
                if stored != params:
                    return False
            except Exception:
                return False
        return True

    def status(self, steps: list[Callable[..., Any]]) -> dict[str, bool]:
        """Return a dict mapping ``fn.__qualname__`` → ``True/False`` indicating
        whether all cached output files exist for each ``@cacheable`` step.
        Non-decorated callables in *steps* are silently skipped.
        """
        result: dict[str, bool] = {}
        for fn in steps:
            info: _CacheInfo | None = getattr(fn, "_cache_info", None)
            if info is None:
                continue
            files = info.get_files(info.step_name)
            result[fn.__qualname__] = all((self.cache_dir / f).exists() for f in files)
        return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_return_type(fn: Callable[..., Any]) -> type | None:
    """Resolve the return-type annotation without calling ``get_type_hints()``.

    ``get_type_hints()`` resolves *all* annotations and raises ``NameError``
    for ``TYPE_CHECKING``-only imports (e.g. ``Dwi`` in ``csd.py``).  We only
    need the return annotation, so we evaluate just that string in the
    function's module namespace.
    """
    ann = fn.__annotations__.get("return")
    if ann is None:
        return None
    if not isinstance(ann, str):
        return ann  # type: ignore[no-any-return]
    module = sys.modules.get(fn.__module__)
    if module is None:
        return None
    try:
        return eval(ann, vars(module))  # type: ignore[no-any-return]
    except Exception:
        return None


def _has_cache_protocol(t: Any) -> bool:
    return (
        callable(getattr(t, "_cache_files", None))
        and callable(getattr(t, "_cache_save", None))
        and callable(getattr(t, "_cache_load", None))
    )


def _extract_scalar_params(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract scalar (``int``, ``float``, ``str``, ``bool``, ``None``) arguments
    from a call for parameter fingerprinting.  Non-scalar args are silently
    skipped.  Returns ``None`` if no scalar args are present.
    """
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except Exception:
        return None
    params = {
        k: v
        for k, v in bound.arguments.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }
    return params or None


def _save_params(
    cache_dir: Path, step_name: str, params: dict[str, Any] | None
) -> None:
    if not params:
        return
    (cache_dir / f"{step_name}.params.json").write_text(
        json.dumps(params, sort_keys=True), encoding="utf-8"
    )


def cacheable(fn_or_spec: Callable[..., Any] | CacheSpec[Any]) -> Any:
    """Decorator that adds transparent caching when a :class:`Cache`
    context is active.  Outside a ``with Cache(...)`` block the
    function runs normally with no caching overhead.

    Two styles:

    - **Bare** ``@cacheable`` — return type must implement the cache protocol
      (``_cache_files``, ``_cache_save``, ``_cache_load``).
    - **Explicit spec** ``@cacheable(CacheSpec(...))`` — for return types that
      cannot carry the protocol (tuples, arrays, etc.).

    Scalar arguments are fingerprinted automatically; a change forces recompute.
    """
    if isinstance(fn_or_spec, CacheSpec):
        spec: CacheSpec[Any] = fn_or_spec

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            step_name = spec.step_name or fn.__name__

            @functools.wraps(fn)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                cache = _active_cache.get()
                if cache is None:
                    return fn(*args, **kwargs)
                params = _extract_scalar_params(fn, args, kwargs)
                if cache.is_cached(step_name, spec.files, params):
                    return spec.load(cache.cache_dir)
                result = fn(*args, **kwargs)
                spec.save(result, cache.cache_dir)
                _save_params(cache.cache_dir, step_name, params)
                return spec.load(cache.cache_dir)

            _wrapper._cache_info = _CacheInfo(  # type: ignore[attr-defined]
                step_name=step_name,
                get_files=lambda _sn: spec.files,
            )
            return _wrapper

        return _decorator

    # Bare @cacheable — fn_or_spec is the function itself.
    # Return-type resolution is deferred to first use so that @cacheable can be
    # stacked directly on @staticmethod inside a class body, where the class
    # name isn't yet in the module namespace at decoration time.
    fn = fn_or_spec
    step_name = fn.__name__
    _resolved_type: Any = None

    def _get_return_type() -> Any:
        nonlocal _resolved_type
        if _resolved_type is None:
            rt = _resolve_return_type(fn)
            if rt is None:
                msg = f"@cacheable: {fn.__qualname__} has no resolvable return type annotation."
                raise TypeError(msg)
            if not _has_cache_protocol(rt):
                msg = (
                    f"@cacheable: return type {rt!r} of {fn.__qualname__} does not "
                    "implement the cache protocol (_cache_files, _cache_save, _cache_load)."
                )
                raise TypeError(msg)
            _resolved_type = rt
        return _resolved_type

    @functools.wraps(fn)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        cache = _active_cache.get()
        if cache is None:
            return fn(*args, **kwargs)
        rt = _get_return_type()
        params = _extract_scalar_params(fn, args, kwargs)
        files = rt._cache_files(step_name)
        if cache.is_cached(step_name, files, params):
            return rt._cache_load(cache.cache_dir, step_name)
        result = fn(*args, **kwargs)
        result._cache_save(cache.cache_dir, step_name)
        _save_params(cache.cache_dir, step_name, params)
        return rt._cache_load(cache.cache_dir, step_name)

    _wrapper._cache_info = _CacheInfo(  # type: ignore[attr-defined]
        step_name=step_name,
        get_files=lambda sn: _get_return_type()._cache_files(sn),
    )
    return _wrapper
