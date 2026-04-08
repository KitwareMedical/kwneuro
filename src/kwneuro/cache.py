from __future__ import annotations

import contextvars
import dataclasses
import functools
import hashlib
import inspect
import json
import sys
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")

_active_cache: contextvars.ContextVar[Cache | None] = contextvars.ContextVar(
    "_active_cache", default=None
)

# Scalar types are stored verbatim as human-readable JSON in the params sidecar.
# All other argument types go through content fingerprinting (see _compute_fingerprint).
_SCALAR_TYPES = (bool, int, float, str, type(None))


@dataclass
class CacheSpec(Generic[T]):
    """Explicit cache spec for ``@cacheable`` when the return type does not implement the cache protocol.

    Use the bare ``@cacheable`` decorator instead when the return type implements
    ``_cache_files``, ``_cache_save``, and ``_cache_load``.
    """

    files: list[str]
    """Output filenames relative to the cache directory."""

    save: Callable[[T, Path], None]
    """Callable that persists the result to the cache directory."""

    load: Callable[[Path], T]
    """Callable that reconstructs the result from the cache directory."""

    step_name: str = ""
    """Step name used to identify this step in the params sidecar. Defaults to the decorated function name."""


@dataclass
class _CacheInfo:
    """Internal metadata attached to each @cacheable-decorated function."""

    step_name: str
    get_files: Callable[[str], list[str]]


@dataclass
class Cache:
    """Context manager that activates caching for all ``@cacheable``-decorated functions.

    Scalar arguments and imaging data arguments are fingerprinted automatically.
    Changes to either trigger a cache miss on the next run. Arguments that cannot
    be fingerprinted issue a UserWarning. Each subject should use a distinct
    cache_dir to avoid races.
    """

    cache_dir: Path
    """Directory where cached outputs and per-step sidecar files are stored. Created automatically."""

    force: bool | set[str] = field(default=False)
    """Cache bypass control. False uses all cached outputs, True recomputes everything,
    a set of step names recomputes only those steps."""

    _token: contextvars.Token[Cache | None] | None = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __enter__(self) -> Cache:
        self._token = _active_cache.set(self)
        return self

    def __exit__(self, *_: Any) -> None:
        if self._token is not None:
            _active_cache.reset(self._token)
            self._token = None

    def is_forced(self, step_name: str) -> bool:
        """Return True if this step should bypass the cache."""
        if isinstance(self.force, set):
            return step_name in self.force
        return bool(self.force)

    def is_cached(
        self,
        step_name: str,
        files: list[str],
        scalars: dict[str, Any] | None = None,
        hashes: dict[str, str] | None = None,
    ) -> bool:
        """Return True when all output files exist, the step is not forced, and the stored fingerprint matches.

        The sidecar file stores two sections: "scalars" for human-readable scalar argument
        values and "hashes" for sha256 content fingerprints. A mismatch in either section,
        or a missing sidecar when params are expected, is treated as a cache miss.

        Args:
            step_name: Name of the cache step, used to locate the sidecar file.
            files: Output filenames relative to cache_dir that must all exist for a cache hit.
            scalars: Scalar argument values to compare against the stored sidecar, or None.
            hashes: Content fingerprints to compare against the stored sidecar, or None.

        Returns: True on a cache hit, False on a miss.
        """
        if self.is_forced(step_name):
            return False
        if not all((self.cache_dir / f).exists() for f in files):
            return False
        if scalars or hashes:
            params_file = self.cache_dir / f"{step_name}.params.json"
            if not params_file.exists():
                return False
            try:
                stored = json.loads(params_file.read_text(encoding="utf-8"))
                # Compare the scalars section (human-readable parameter values).
                if scalars and stored.get("scalars") != scalars:
                    return False
                # Compare the hashes section (content fingerprints for imaging data).
                if hashes and stored.get("hashes") != hashes:
                    return False
            except Exception:  # pylint: disable=broad-exception-caught
                return False
        return True

    def status(self, steps: list[Callable[..., Any]]) -> dict[str, bool]:
        """Return a mapping of each step's qualified name to whether all its cached output files exist.

        Non-decorated callables in steps are silently skipped.

        Args:
            steps: List of cacheable-decorated functions to check.

        Returns: Dict mapping fn.__qualname__ to True if all cached outputs exist, False otherwise.
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
# Fingerprinting
# ---------------------------------------------------------------------------
# The fingerprinting system converts any argument value into a stable sha256
# hex string so that changes to imaging data — not just scalar parameters —
# trigger a cache miss.  All logic lives here; no cache-related methods are
# added to resource or data classes.


def _compute_fingerprint(v: Any) -> str | None:
    """Compute a stable sha256 fingerprint string for v, or None if v cannot be fingerprinted.

    Handles scalars, numpy arrays, numpy scalars, Path, dict, list/tuple, and dataclass
    instances recursively. Dataclass support covers the full resource hierarchy (Dwi,
    VolumeResource, etc.) without importing those types. Returns None for unrecognised
    types; the caller issues a UserWarning in that case.

    Args:
        v: The value to fingerprint.

    Returns: A sha256 hex digest string, or None if v cannot be fingerprinted.
    """
    h = hashlib.sha256()

    # --- Scalars ---
    # bool must be checked before int because bool is a subclass of int.
    if v is None:
        h.update(b"None:")
        return h.hexdigest()
    if isinstance(v, bool):
        h.update(f"bool:{v}".encode())
        return h.hexdigest()
    if isinstance(v, int):
        h.update(f"int:{v}".encode())
        return h.hexdigest()
    if isinstance(v, float):
        # Use repr for floats to preserve full precision.
        h.update(f"float:{v!r}".encode())
        return h.hexdigest()
    if isinstance(v, str):
        h.update(f"str:{v}".encode())
        return h.hexdigest()

    # --- numpy arrays ---
    # Hash raw bytes plus shape and dtype so arrays with the same bytes but
    # different shapes (e.g. (2,3) vs (6,)) produce different fingerprints.
    if isinstance(v, np.ndarray):
        h.update(v.tobytes())
        h.update(f"|shape={v.shape}|dtype={v.dtype}".encode())
        return h.hexdigest()

    # --- numpy scalar types (np.float32, np.int64, etc.) ---
    if isinstance(v, np.generic):
        h.update(f"npscalar:{type(v).__name__}:{v!r}".encode())
        return h.hexdigest()

    # --- Path objects ---
    # Disk resources (NiftiVolumeResource, FslBvalResource, etc.) store their
    # data as a Path field, so this gives them a path-identity fingerprint.
    if isinstance(v, Path):
        h.update(f"path:{v}".encode())
        return h.hexdigest()

    # --- Dicts ---
    # Sort by key string for determinism across runs.
    if isinstance(v, dict):
        parts = []
        for k, dv in sorted(v.items(), key=lambda item: str(item[0])):
            fp = _compute_fingerprint(dv)
            if fp is None:
                return None  # a dict value is untrackable → whole dict is untrackable
            parts.append(f"{k}:{fp}")
        h.update("|".join(parts).encode())
        return h.hexdigest()

    # --- Lists and tuples ---
    if isinstance(v, (list, tuple)):
        parts = []
        for item in v:
            fp = _compute_fingerprint(item)
            if fp is None:
                return None
            parts.append(fp)
        h.update("|".join(parts).encode())
        return h.hexdigest()

    # --- Dataclass instances ---
    # Covers Dwi, InMemoryVolumeResource, NiftiVolumeResource, InMemoryBvalResource,
    # InMemoryBvecResource, InMemoryResponseFunctionResource, and any future
    # resource types, as long as their fields are among the supported types above.
    # The `not isinstance(v, type)` guard excludes dataclass *classes* themselves.
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        parts = []
        for f in dataclasses.fields(v):
            fp = _compute_fingerprint(getattr(v, f.name))
            if fp is None:
                return None  # an untrackable field → whole dataclass is untrackable
            parts.append(f"{f.name}:{fp}")
        h.update("|".join(parts).encode())
        return h.hexdigest()

    # Unrecognised type — the caller will warn that this argument is not tracked.
    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_return_type(fn: Callable[..., Any]) -> type | None:
    """Resolve the return-type annotation for fn without calling get_type_hints().

    get_type_hints() raises NameError for TYPE_CHECKING-only imports (e.g. Dwi in csd.py).
    Evaluates only the return annotation string in the function's own module namespace.
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
        return eval(ann, vars(module))  # type: ignore[no-any-return]  # pylint: disable=eval-used
    except Exception:  # pylint: disable=broad-exception-caught
        return None


def _has_cache_protocol(t: Any) -> bool:
    return (
        callable(getattr(t, "_cache_files", None))
        and callable(getattr(t, "_cache_save", None))
        and callable(getattr(t, "_cache_load", None))
    )


def _extract_params(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, str] | None]:
    """Classify bound arguments into scalars, content fingerprints, or untracked.

    Scalar arguments (bool, int, float, str, None) are stored verbatim. Fingerprint-able
    arguments (dataclasses, numpy arrays, Path, etc.) are stored as sha256 hex digests.
    Untracked arguments trigger a UserWarning.

    Args:
        fn: The wrapped function, used for signature binding and warning messages.
        args: Positional arguments from the call.
        kwargs: Keyword arguments from the call.

    Returns: Tuple of (scalars dict or None, hashes dict or None).
    """
    try:
        sig = inspect.signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
    except Exception:  # pylint: disable=broad-exception-caught
        return None, None

    scalars: dict[str, Any] = {}
    hashes: dict[str, str] = {}

    for k, v in bound.arguments.items():
        if isinstance(v, _SCALAR_TYPES):
            # Scalar — store as a human-readable JSON value (e.g. dpar=1.7e-3).
            scalars[k] = v
        else:
            fp = _compute_fingerprint(v)
            if fp is not None:
                # Content-fingerprint-able (e.g. Dwi, VolumeResource).
                # Any change to the underlying data will produce a different
                # sha256 and cause a cache miss on the next run.
                hashes[k] = fp
            else:
                # Cannot fingerprint — warn so the user knows this argument
                # is invisible to the cache and won't trigger recomputation.
                warnings.warn(
                    f"@cacheable: parameter '{k}' of type '{type(v).__name__}' "
                    f"in '{fn.__qualname__}' cannot be fingerprinted and will not "
                    "be tracked for cache invalidation. "
                    f"Use force={{'{fn.__name__}'}} to force recomputation "
                    "when this input changes.",
                    UserWarning,
                    stacklevel=4,
                )

    return scalars or None, hashes or None


def _save_params(
    cache_dir: Path,
    step_name: str,
    scalars: dict[str, Any] | None,
    hashes: dict[str, str] | None,
) -> None:
    """Write scalar params and content hashes to {step_name}.params.json.

    The sidecar stores two sections: "scalars" for human-readable scalar argument values
    and "hashes" for sha256 content fingerprints. No file is written if both are empty.

    Args:
        cache_dir: Directory where the sidecar file is written.
        step_name: Used to name the sidecar file ({step_name}.params.json).
        scalars: Scalar argument values to persist, or None.
        hashes: Content fingerprints to persist, or None.
    """
    if not scalars and not hashes:
        return
    data: dict[str, Any] = {}
    if scalars:
        data["scalars"] = scalars
    if hashes:
        data["hashes"] = hashes
    (cache_dir / f"{step_name}.params.json").write_text(
        json.dumps(data, sort_keys=True), encoding="utf-8"
    )


def cacheable(fn_or_spec: Callable[..., Any] | CacheSpec[Any]) -> Any:
    """Decorator that adds transparent caching when a Cache context is active.

    Outside a ``with Cache(...)`` block the function runs normally with no overhead.
    Scalar and dataclass arguments are fingerprinted automatically; a change in either
    causes a cache miss. Arguments that cannot be fingerprinted trigger a UserWarning.

    Can be used in two ways. As a bare decorator when the return type implements
    the cache protocol (_cache_files, _cache_save, _cache_load)::

        @cacheable
        def estimate_dti(dwi: Dwi, mask: VolumeResource | None = None) -> Dti: ...

    Or with an explicit CacheSpec for return types that cannot carry the protocol::

        @cacheable(CacheSpec(files=[...], save=..., load=...))
        def compute_csd_peaks(...) -> tuple[VolumeResource, VolumeResource]: ...
    """
    # The cache protocol methods (_cache_files, _cache_save, _cache_load, _cache_info)
    # are prefixed with _ to mark them as internal to the caching infrastructure, not
    # as truly protected class members. cacheable() is the intended call site for all of them.
    # pylint: disable=protected-access
    if isinstance(fn_or_spec, CacheSpec):
        spec: CacheSpec[Any] = fn_or_spec

        def _decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
            step_name = spec.step_name or fn.__name__

            @functools.wraps(fn)
            def _wrapper(*args: Any, **kwargs: Any) -> Any:
                cache = _active_cache.get()
                if cache is None:
                    # No active Cache context — run the function as-is.
                    return fn(*args, **kwargs)
                # Classify arguments into scalars and content hashes.
                scalars, hashes = _extract_params(fn, args, kwargs)
                if cache.is_cached(step_name, spec.files, scalars, hashes):
                    # All output files exist and the fingerprint matches — load from disk.
                    return spec.load(cache.cache_dir)
                # Cache miss: run the function, persist the outputs, save the fingerprint.
                result = fn(*args, **kwargs)
                spec.save(result, cache.cache_dir)
                _save_params(cache.cache_dir, step_name, scalars, hashes)
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
            # No active Cache context — run the function as-is.
            return fn(*args, **kwargs)
        rt = _get_return_type()
        # Classify arguments into scalars and content hashes.
        scalars, hashes = _extract_params(fn, args, kwargs)
        files = rt._cache_files(step_name)
        if cache.is_cached(step_name, files, scalars, hashes):
            # All output files exist and the fingerprint matches — load from disk.
            return rt._cache_load(cache.cache_dir, step_name)
        # Cache miss: run the function, persist the output, save the fingerprint.
        result = fn(*args, **kwargs)
        result._cache_save(cache.cache_dir, step_name)
        _save_params(cache.cache_dir, step_name, scalars, hashes)
        return rt._cache_load(cache.cache_dir, step_name)

    _wrapper._cache_info = _CacheInfo(  # type: ignore[attr-defined]
        step_name=step_name,
        get_files=lambda sn: _get_return_type()._cache_files(sn),
    )
    return _wrapper
