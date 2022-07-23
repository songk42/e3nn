import collections

import itertools
import numpy as np
import torch
from e3nn import o3
from e3nn.math import germinate_formulas, orthonormalize, reduce_permutation
from e3nn.util import explicit_default_types
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from torch import fix

def _subformulas(f0, formulas, subset):
    subset_indices = {i for i in range(len(f0)) if f0[i] in subset}
    standard_indices = {ndx: i for i, ndx in enumerate(subset)}

    subformulas_st = set()
    for s, f in formulas:
        if all([f0[i] in subset or f[i] == i for i in range(len(f0))]):
            f_filtered = tuple(filter(lambda x: x in subset_indices, f))
            f_standard = tuple(map(lambda x: standard_indices[f0[x]], f_filtered))
            subformulas_st.add((s, f_standard))

    return subformulas_st


def _find_P_dim(f0, formulas, dtype=None, device=None, **dims):
    # here we check that each index has one and only one dimension
    for _s, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in dims and j in dims and dims[i] != dims[j]:
                raise RuntimeError(f'dimension of {i} and {j} should be the same')
            if i in dims:
                dims[j] = dims[i]
            if j in dims:
                dims[i] = dims[j]

    for i in f0:
        if i not in dims:
            raise RuntimeError(f'index {i} has no dimension associated to it')

    dims = [dims[i] for i in f0]

    full_base = list(itertools.product(*(range(d) for d in dims))) 

    base = set()
    for x in full_base:
        # T[x] is a coefficient of the tensor T and is related to other coefficient T[y]
        # if x and y are related by a formula
        xs = {(s, tuple(x[i] for i in p)) for s, p in formulas}
        # s * T[x] are all equal for all (s, x) in xs
        # if T[x] = -T[x] it is then equal to 0 and we lose this degree of freedom
        if not (-1, x) in xs:
            # the sign is arbitrary, put both possibilities
            base.add(frozenset({
                frozenset(xs),
                frozenset({(-s, x) for s, x in xs})
            }))

    prod = len(base)
    for i in dims:
        prod *= i
    return prod
