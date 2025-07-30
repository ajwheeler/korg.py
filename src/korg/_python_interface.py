"""
This module defines the python interface to Korg
"""

import os
from collections import ChainMap
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Never, TypedDict
from ._julia_import import jl, Korg

import numpy as np
from juliacall import VectorValue as jlVectorValue

# define some type aliases

# this is a placeholder annotation that we should delete
type ToBeAnnotated = Any

# this is a placeholder for arguments in a julia function that can't currently
# be represented in Python
type NoPyType = Never

# this is placeholder for float-like objects. In addition to Python's float type,
# it also allows numpy scalar types.
# - Due to a weird python typing rule, the fact that we permit float means that we
#   also permit int
# - typing.SupportsFloat is much more limited (its only useful for forwarding or if you
#   plan to immediately call float(arg) and subsequently work the result of that call,
#   rather that arg
type KFloat = float | np.float32 | np.float64

# this is a placeholder for family of types that can be coerced to Korg.jl's internal
# Wavelengths Type
# Todo: get more explicit!
type WavelengthsType = Any


type Array1dF64 = np.ndarray[tuple[int], np.dtype[np.float64]]


def _perfect_jl_shadowing[**P, T](fn: Callable[P, T]) -> Callable[P, T]:
    """A decorator for functions that perfectly shadows a Korg

    The main purpose of this function is to add the Korg docstring. This should
    be used somewhat sparingly (i.e. in cases when we are confident that the
    docstring will never mention Julia-specific types)
    """
    _recycle_jl_docstring(fn)
    return fn


def _recycle_jl_docstring(fn: Callable):
    # this is experimental (to be used sparingly in cases when we are confident that
    # the docstrings won't mention Julia specific types)
    #
    # this is separate from _perfect_jl_shadowing because there may be a lot of heavy
    # lifting (and there could conceivably be cases where we want to reuse part of a
    # docstring)

    # TODO: we need to figure out how to best translate Documenter.jl flavored markdown
    #       to restructured text. Since the docstrings of all public Korg functions
    #       largely share a common structure, it probably wouldn't be bad to do this
    #       with a few regex statements
    jl_docstring = jl.seval(f"(@doc Korg.{fn.__name__}).text[1]")

    if jl_docstring.startswith(f"    {fn.__name__}("):
        first_newline = jl_docstring.index("\n")
        fn.__doc__ = jl_docstring[first_newline:].lstrip()
    else:
        raise RuntimeError(
            f"There was a problem getting the docstring for {fn.__name__}"
        )


class Linelist:
    """A lightweight class that wraps a linelist.

    You shouldn't try to initialize this class directly. Instead, you should rely upon
    functions like :py:func:`~korg.get_APOGEE_DR17_linelist`,
    :py:func:`~korg.get_GES_linelist`, etc.
    """

    _lines: jlVectorValue

    def __init__(self, lines: jlVectorValue):
        # this is NOT a public method
        self._lines = lines

    def __len__(self) -> int:
        return len(self._lines)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        length = len(self)
        return f"{name}(<{length} lines>)"


@_perfect_jl_shadowing
def get_APOGEE_DR17_linelist(*, include_water: bool = True) -> Linelist:
    return Linelist(Korg.get_APOGEE_DR17_linelist(include_water=include_water))


@_perfect_jl_shadowing
def get_GALAH_DR3_linelist() -> Linelist:
    return Linelist(Korg.get_GALAH_DR3_linelist())


@_perfect_jl_shadowing
def get_GES_linelist(*, include_molecules: bool = True) -> Linelist:
    return Linelist(Korg.get_GES_linelist(include_molecules=include_molecules))


@_perfect_jl_shadowing
def get_VALD_solar_linelist() -> Linelist:
    return Linelist(Korg.get_VALD_solar_linelist())


# we can't currently reuse the exact Julia signature since the Julia signature
# explicitly states that it returns a vector of Lines
def read_linelist(
    fname: os.PathLike,
    *,
    format: str | None = None,
    isotopic_abundances: Mapping[int, Mapping[float, float]] | None = None,
) -> Linelist:
    # coerce fname to a string
    coerced_fname = os.fsdecode(fname)

    # build up kwargs (we have to play some games here since we can't natively
    # represent the default values in python)
    kwargs: dict[str, Any] = {}
    if format is not None:
        kwargs["format"] = format
    if isotopic_abundances is not None:
        kwargs["isotopic_abundances"] = isotopic_abundances
    return Linelist(Korg.read_linelist(coerced_fname, **kwargs))


class SynthesizeTypedKwargDict(TypedDict, total=False):
    """
    Provides type annotations for the dict of optional kwargs that are forwarded to
    the `synthesize` Julia Function provided by Korg.jl

    This type assumes that if you want to use a keyword argument's default value, you
    simply won't pass that keyword argument. Any annotation of `None` means that the
    julia keyword argument accepts `nothing`. This type is mostly intended for
    type-checking purposes with static analysis tools like mypy.
    """

    vmic: KFloat
    line_buffer: KFloat
    cntm_step: KFloat
    air_wavelengths: bool
    hydrogen_lines: bool
    use_MHD_for_hydrogen_lines: bool
    hydrogen_line_window_size: int
    mu_values: int
    line_cutoff_threshold: KFloat
    electron_number_density_warn_threshold: KFloat
    electron_number_density_warn_min_value: KFloat
    return_cntm: bool
    I_scheme: str
    tau_scheme: str
    ionization_energies: ToBeAnnotated
    partition_funcs: ToBeAnnotated
    log_equilibrium_constants: ToBeAnnotated
    molecular_cross_sections: ToBeAnnotated
    # the ONLY reason that the next entry expects None is because the Julia function
    # may accept `nothing` (the julia counterpart for None)
    use_chemical_equilibrium_from: NoPyType | None
    verbose: bool


class FormatAXTypedKwargDict(TypedDict, total=False):
    """
    Provides type annotations for the dict of optional kwargs that are forwarded to
    the `format_A_X` Julia Function provided by Korg.jl

    This type assumes that if you want to use a keyword argument's default value, you
    simply won't pass that keyword argument. Any annotation of `None` means that the
    julia keyword argument accepts `nothing`. This type is mostly intended for
    type-checking purposes with static analysis tools like mypy.
    """

    solar_relative: bool
    solar_abundances: Sequence[KFloat]
    alpha_elements: Sequence[int]


class SynthResult(NamedTuple):
    wls: Array1dF64
    flux: Array1dF64
    continuum: Array1dF64


# we can't currently reuse the exact Julia signature since the Julia signature
# explicitly references synthesize and format_A_X, which we are not providing python
# wrappers for at this time
def synth(
    *,
    Teff: KFloat,
    logg: KFloat,
    m_H: KFloat = 0.0,
    alpha_H: KFloat | None = None,
    linelist: Linelist | None = None,
    wavelengths: WavelengthsType = (5000, 6000),
    rectify: bool = True,
    R: KFloat = float("inf"),
    vsini: KFloat = 0.0,
    vmic: KFloat = 1.0,
    synthesize_kwargs: SynthesizeTypedKwargDict | None = None,
    format_A_X_kwargs: FormatAXTypedKwargDict | None = None,
    # the following annotation applies to all kwarg values
    **abundances: KFloat,
) -> SynthResult:
    """Creates a synthetic spectrum... TODO: ADD A FULL DOCSTRING"""

    # here, we deal with building up a subset of the keyword arguments where we use
    # None to indicate that we can't represent the default value in python

    partial_kwargs: dict[str, Any] = {}
    if alpha_H is not None:
        partial_kwargs["alpha_H"] = alpha_H
    if linelist is not None:
        partial_kwargs["linelist"] = linelist._lines
    if synthesize_kwargs is not None:
        partial_kwargs["synthesize_kwargs"] = synthesize_kwargs
    if format_A_X_kwargs is not None:
        partial_kwargs["format_A_X_kwargs"] = format_A_X_kwargs

    tmp_wls, tmp_flux, tmp_continuum = Korg.synth(
        Teff=Teff,
        logg=logg,
        m_H=m_H,
        wavelengths=wavelengths,
        rectify=rectify,
        R=R,
        vsini=vsini,
        vmic=vmic,
        **ChainMap(partial_kwargs, abundances),
    )

    # we are returning numpy arrays that wrap each of the Julia vectors
    # -> because `juliacall.VectorValue` subclasses `juliacall.ArrayValue`, and
    #    `juliacall.ArrayValue` properly implements the `obj.__array_interface__`
    #    python property, `arr = np.array(vec, copy=False)` makes `arr` reference
    #    a ndarray instance that
    #    - reuses the memory of the underlying julia vector tracked within the python
    #      object that `vec` references
    #    - properly tracks a reference to the python variable `vec` references. Thus:
    #      - if `vec` goes out of scope before `arr`, the object previously referenced
    #        by `vec` will be kept alive by a reference held in `arr`.
    #      - a reference to the python object holding the julia vector is obviously
    #        stored in any other numpy arrays that are created that view the underlying
    #        memory referenced by `arr`
    #      - you can see this reference by looking at `arr.base`. At the time of
    #        writing, `arr.base` technically references a `memoryview` that references
    #        `juliacall.VectorValue` python object. You can see this by looking at
    #        `arr.base.obj`
    # -> It's **ALMOST CERTAINLY** a really good thing that we essentially "forget"
    #    that these buffers are implemented as mutable vectors as we convert them to
    #    numpy arrays. Problems would probably arise if we changed vector capacity
    #    (i.e. we started adding items):
    #    - AFAIK, the only plausible way to provide an arbitrary capacity vector with
    #      contiguous memory, is to realloc when we need more capacity. The new memory
    #      allocation is probably at a different memory address and I'm pretty sure
    #      that there isn't machinery to communicate the change to numpy/python.
    #      Consequently the numpy array will hold a dangling pointer.
    #    - even if the pointer to the reallocated memory kept the same address, I
    #      suspect that pointer provenance issues could hypothetically arise...
    #    - this is something that comes up a lot when using C++'s std::vector
    # -> It's worth noting that it would be better if we could just pass the output
    #    buffers directly to Korg.jl.
    #    - But, that would realistically take a lot of work and this solution is
    #      probably "good enough." This approach is better than doing a memcpy and is
    #      probably only an issue when memory is limited
    #    - The limitation here relates to "how delayed" memory freeing is... When the
    #      last reference to a numpy array goes out of scope, we need to:
    #      1. wait for the python garbage collector to run (& reduce the ref count for
    #         the memory view), garbage collect the memory view (& reduce the ref count
    #         for the `juliacall.VectorValue` python object, and collect
    #         `juliacall.VectorValue` python object. (I guess it's plausible this
    #         could be completed in a single step)
    #      2. we need julia to then deallocate its memory
    return SynthResult(
        wls=np.array(tmp_wls, copy=False),
        flux=np.array(tmp_flux, copy=False),
        continuum=np.array(tmp_continuum, copy=False),
    )
