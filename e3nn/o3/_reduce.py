import collections

import torch
import itertools
from e3nn import o3
from e3nn.math import germinate_formulas, orthonormalize, reduce_permutation
from e3nn.util import explicit_default_types
from e3nn.util.codegen import CodeGenMixin
from e3nn.util.jit import compile_mode
from torch import fx
from ._reduce_util import _find_P_dim, _subformulas
from e3nn.o3._wigner import wigner_3j

_TP = collections.namedtuple("tp", "op, args")
_INPUT = collections.namedtuple("input", "tensor, start, stop")


def _wigner_nj(*irrepss, normalization="component", filter_ir_mid=None, dtype=None, device=None):
    irrepss = [o3.Irreps(irreps) for irreps in irrepss]
    if filter_ir_mid is not None:
        filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]

    if len(irrepss) == 1:
        (irreps,) = irrepss
        ret = []
        e = torch.eye(irreps.dim, dtype=dtype, device=device)
        i = 0
        for mul, ir in irreps:
            for _ in range(mul):
                sl = slice(i, i + ir.dim)
                ret += [(ir, _INPUT(0, sl.start, sl.stop), e[sl])]
                i += ir.dim
        return ret

    *irrepss_left, irreps_right = irrepss
    ret = []
    for ir_left, path_left, C_left in _wigner_nj(
        *irrepss_left, normalization=normalization, filter_ir_mid=filter_ir_mid, dtype=dtype, device=device
    ):
        i = 0
        for mul, ir in irreps_right:
            for ir_out in ir_left * ir:
                if filter_ir_mid is not None and ir_out not in filter_ir_mid:
                    continue

                C = o3.wigner_3j(ir_out.l, ir_left.l, ir.l, dtype=dtype, device=device)
                if normalization == "component":
                    C *= ir_out.dim ** 0.5
                if normalization == "norm":
                    C *= ir_left.dim ** 0.5 * ir.dim ** 0.5

                C = torch.einsum("jk,ijl->ikl", C_left.flatten(1), C)
                C = C.reshape(ir_out.dim, *(irreps.dim for irreps in irrepss_left), ir.dim)
                for u in range(mul):
                    E = torch.zeros(
                        ir_out.dim, *(irreps.dim for irreps in irrepss_left), irreps_right.dim, dtype=dtype, device=device
                    )
                    sl = slice(i + u * ir.dim, i + (u + 1) * ir.dim)
                    E[..., sl] = C
                    ret += [
                        (
                            ir_out,
                            _TP(op=(ir_left, ir, ir_out), args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop))),
                            E,
                        )
                    ]
            i += mul * ir.dim

    return sorted(ret, key=lambda x: x[0])


def find_R(irreps1, irreps2, Q1, Q2, paths_left, I, filter_ir_out=None, normalization='component', dtype=None):
    Rs = {}  # dictionary of irreps -> matrix
    irreps_out = []
    k1 = 0
    for mul1, ir1 in irreps1:
        sub_Q1 = Q1[k1:k1 + mul1 * ir1.dim].reshape(mul1, ir1.dim, -1)
        k2 = 0
        for mul2, ir2 in irreps2:
            sub_Q2 = Q2[k2:k2 + mul2 * ir2.dim].reshape(mul2, ir2.dim, -1)
            for ir_out in ir1 * ir2:
                C = wigner_3j(ir1.l, ir2.l, ir_out.l, dtype=dtype)
                if normalization == 'component':
                    C *= ir_out.dim**0.5
                if normalization == 'norm':
                    C *= ir1.dim**0.5 * ir2.dim**0.5
                C = torch.einsum("mia,njb,ijk->mnkab", sub_Q1, sub_Q2, C)
                C = C.reshape(mul1 * mul2, C.shape[2], *(Q1.shape[1:]), *(Q2.shape[1:]))
                if filter_ir_out is None or ir_out in filter_ir_out:
                    irreps_out.append((mul1 * mul2, ir_out))
                    # path is something like
                    # _TP(
                    #     op=(ir_left, ir, ir_out),
                    #     args=(path_left, _INPUT(len(irrepss_left), sl.start, sl.stop))
                    # )
                    # i + u * ir.dim, i + (u + 1) * ir.dim
                    if ir_out not in Rs.keys():
                        Rs[ir_out] = []
                    for i1, path_left in enumerate(paths_left[ir1]):
                        for i2 in range(mul2):
                            path = _TP(
                                op=(ir1, ir2, ir_out),
                                args=(path_left, _INPUT(I, k2 + i2 * ir2.dim, k2 + (i2 + 1) * ir2.dim))
                            )
                            Rs[ir_out].append((path, C[i1 * mul2 + i2]))
            k2 += mul2 * ir2.dim
        k1 += mul1 * ir1.dim
    return o3.Irreps(sorted(irreps_out)).simplify(), Rs


def find_Q(P, Rs, eps=1e-9, dtype=None):
    Q = []
    outputs = []
    paths_out = {}
    irreps_out = []
    PP = P @ P.T  # (a,a)

    for ir in Rs:
        mul = len(Rs[ir])
        paths = [path for path, _ in Rs[ir]]
        base_o3 = torch.stack([R for _, R in Rs[ir]])

        R = base_o3.flatten(2)  # [multiplicity, ir, input basis] (u,j,omega)

        proj_s = []  # list of projectors into vector space
        for j in range(ir.dim):
            # Solve X @ R[:, j] = Y @ P, but keep only X
            RR = R[:, j] @ R[:, j].T  # (u,u)
            RP = R[:, j] @ P.T  # (u,a)

            prob = torch.cat([
                torch.cat([RR, -RP], dim=1),
                torch.cat([-RP.T, PP], dim=1)
            ], dim=0)
            eigenvalues, eigenvectors = torch.linalg.eigh(prob)
            X = eigenvectors[:, eigenvalues < eps][:mul].T  # [solutions, multiplicity]
            proj_s.append(X.T @ X)

            break  # do not check all components because too time expensive

        for p in proj_s:
            assert (p - proj_s[0]).abs().max() < eps, f"found different solutions for irrep {ir}"

        # look for an X such that X.T @ X = Projector
        X, _ = orthonormalize(proj_s[0], eps)

        paths_out_tmp = []
        for x in X:
            C = torch.einsum("u,ui...->i...", x, base_o3)
            correction = (ir.dim / C.pow(2).sum()) ** 0.5
            C = correction * C

            outputs_tmp = []
            for v, p in zip(x, paths):
                if v.abs() > eps:
                    outputs_tmp.append(((correction * v).item(), p))
                    paths_out_tmp.append(p)
            outputs.append(outputs_tmp)
            Q.append(C)
            irreps_out.append((1, ir)) # not sure if that 1 is right
        if len(paths_out_tmp):
            paths_out[ir] = paths_out_tmp

    irreps_out = o3.Irreps(irreps_out).simplify()
    Q = torch.cat(Q).to(dtype=dtype)
    return irreps_out, Q, outputs, paths_out


def _rtp_dq(f0, formulas, irreps, filter_ir_out=None, filter_ir_mid=None, eps=1e-4, dtype=None):
    # def _rtp_dq(f0, formulas, irreps, filter_ir_out=None, filter_ir_mid=None, eps=1e-9):
    '''irreps: dict of indices to irreps'''
    # base case
    if len(f0) == 1:
        irreps_out = o3.Irreps(irreps[f0[0]])
        output = []
        paths = {}
        k = 0
        for mul, ir in irreps_out:
            for _ in range(mul):
                output.append([(1.0, _INPUT(0, k, k + ir.dim))])
                if ir not in paths:
                    paths[ir] = []
                paths[ir].append(_INPUT(0, k, k + ir.dim))
                k += ir.dim
        return irreps_out, irreps_out, torch.eye(irreps_out.dim), output, paths

    for _sign, p in formulas:
        f = "".join(f0[i] for i in p)
        for i, j in zip(f0, f):
            if i in irreps and j in irreps and irreps[i] != irreps[j]:
                raise RuntimeError(f"irreps of {i} and {j} should be the same")
            if i in irreps:
                irreps[j] = irreps[i]
            if j in irreps:
                irreps[i] = irreps[j]

    for i in f0:
        if i not in irreps:
            raise RuntimeError(f"index {i} has no irreps associated to it")

    for i in irreps:
        if i not in f0:
            raise RuntimeError(f"index {i} has an irreps but does not appear in the fomula")

    # find optimal subformulas
    best_subindices = None
    D_curr = -1
    for subindices in _subsets(len(f0)):
        if len(subindices) > 0 and len(subindices) < len(f0):
            f1 = [f0[i] for i in subindices]
            f2 = [f0[i] for i in range(len(f0)) if i not in subindices]
            formulas1 = _subformulas(f0, formulas, f1)
            formulas2 = _subformulas(f0, formulas, f2)

            p1 = _find_P_dim(f1, formulas1, **{i: irreps[i].dim for i in f1})
            p2 = _find_P_dim(f2, formulas2, **{i: irreps[i].dim for i in f2})
            if p1 * p2 < D_curr or D_curr == -1:
                D_curr = p1 * p2
                best_subindices = subindices
    assert D_curr != -1
    f2 = [f0[i] for i in best_subindices]
    f1 = [f0[i] for i in range(len(f0)) if i not in best_subindices]
    formulas1 = _subformulas(f0, formulas, f1)
    formulas2 = _subformulas(f0, formulas, f2)

    # bases from the full problem
    # permutation basis
    base_perm, _ = reduce_permutation(f0, formulas, dtype, **{i: irs.dim for i, irs in irreps.items()})  # same size as output
    P = base_perm.flatten(1)  # [permutation basis, input basis] (a,omega)

    # Qs from subproblems (irrep outputs)
    _, out1, Q1, outputs1, paths1 = _rtp_dq(f1, formulas1, {c: irreps[c] for c in f1}, filter_ir_out, filter_ir_mid, eps)
    _, out2, Q2, outputs2, _ = _rtp_dq(f2, formulas2, {c: irreps[c] for c in f2}, filter_ir_out, filter_ir_mid, eps)
    assert len(paths1) == len(out1) # not necessarily the case

    irreps_out_R, Rs = find_R(out1, out2, Q1, Q2, paths1, len(f1), filter_ir_out, dtype=dtype)

    # if all symmetries are already accounted for, find_Q isn't necessary
    # R needs to be turned into an array
    # if P.shape[0] == sum(map(lambda v: len(v), Rs.values())):
    #     paths_out = {}
    #     for ir in Rs:
    #         paths_out[ir] = [path for path, _ in Rs[ir]]
    #         base_o3 = torch.stack([R for _, R in Rs[ir]])
    #     return irreps_in, irreps_out, R

    # otherwise, take extra global symmetries into account
    irreps_out, Q, outputs, paths = find_Q(P, Rs, eps, dtype=dtype)
    irreps_in = [irreps[i] for i in f0]
    return irreps_in, irreps_out, Q, outputs, paths  # this "outputs" is _just_ the outputs from the last find_Q


def _subsets(n):
    s = list(range(n))
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(n + 1))


def _get_ops(path):
    if isinstance(path, _INPUT):
        return
    assert isinstance(path, _TP)
    yield path.op
    for op in _get_ops(path.args[0]):
        yield op


@compile_mode("trace")
class ReducedTensorProducts(CodeGenMixin, torch.nn.Module):
    r"""reduce a tensor with symmetries into irreducible representations

    Parameters
    ----------
    formula : str
        String made of letters ``-`` and ``=`` that represent the indices symmetries of the tensor.
        For instance ``ij=ji`` means that the tensor has to indices and if they are exchanged, its value is the same.
        ``ij=-ji`` means that the tensor change its sign if the two indices are exchanged.

    filter_ir_out : list of `e3nn.o3.Irrep`, optional
        Optional, list of allowed irrep in the output

    filter_ir_mid : list of `e3nn.o3.Irrep`, optional
        Optional, list of allowed irrep in the intermediary operations

    **kwargs : dict of `e3nn.o3.Irreps`
        each letter present in the formula has to be present in the ``irreps`` dictionary, unless it can be inferred by the
        formula. For instance if the formula is ``ij=ji`` you can provide the representation of ``i`` only:
        ``ReducedTensorProducts('ij=ji', i='1o')``.

    Attributes
    ----------
    irreps_in : list of `e3nn.o3.Irreps`
        input representations

    irreps_out : `e3nn.o3.Irreps`
        output representation

    change_of_basis : `torch.Tensor`
        tensor of shape ``(irreps_out.dim, irreps_in[0].dim, ..., irreps_in[-1].dim)``

    Examples
    --------
    >>> tp = ReducedTensorProducts('ij=-ji', i='1o')
    >>> x = torch.tensor([1.0, 0.0, 0.0])
    >>> y = torch.tensor([0.0, 1.0, 0.0])
    >>> tp(x, y) + tp(y, x)
    tensor([0., 0., 0.])

    >>> tp = ReducedTensorProducts('ijkl=jikl=ikjl=ijlk', i="1e")
    >>> tp.irreps_out
    1x0e+1x2e+1x4e

    >>> tp = ReducedTensorProducts('ij=ji', i='1o')
    >>> x, y = torch.randn(2, 3)
    >>> a = torch.einsum('zij,i,j->z', tp.change_of_basis, x, y)
    >>> b = tp(x, y)
    >>> assert torch.allclose(a, b, atol=1e-3, rtol=1e-3)
    """
    # pylint: disable=abstract-method

    def __init__(self, formula, filter_ir_out=None, filter_ir_mid=None, eps=1e-4, **irreps):
        super().__init__()

        if filter_ir_out is not None:
            try:
                filter_ir_out = [o3.Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of e3nn.o3.Irrep")

        if filter_ir_mid is not None:
            try:
                filter_ir_mid = [o3.Irrep(ir) for ir in filter_ir_mid]
            except ValueError:
                raise ValueError(f"filter_ir_mid (={filter_ir_mid}) must be an iterable of e3nn.o3.Irrep")

        f0, formulas = germinate_formulas(formula)

        irreps = {i: o3.Irreps(irs) for i, irs in irreps.items()}

        for i in irreps:
            if len(i) != 1:
                raise TypeError(f"got an unexpected keyword argument '{i}'")

        irreps_in, irreps_out, change_of_basis, outputs, _ = _rtp_dq(f0, formulas, irreps, filter_ir_out, filter_ir_mid, eps, dtype=torch.float32)

        dtype, _ = explicit_default_types(None, None)
        self.register_buffer("change_of_basis", change_of_basis.to(dtype=dtype))

        tps = set()
        for vp_list in outputs:
            for v, p in vp_list:
                for op in _get_ops(p):
                    tps.add(op)

        root = torch.nn.Module()

        tps = list(tps)
        for i, op in enumerate(tps):
            tp = o3.TensorProduct(op[0], op[1], op[2], [(0, 0, 0, "uuu", False)])
            setattr(root, f"tp{i}", tp)

        graph = fx.Graph()
        tracer = torch.fx.proxy.GraphAppendingTracer(graph)
        inputs = [fx.Proxy(graph.placeholder(f"x{i}", torch.Tensor), tracer) for i in f0]

        self.irreps_in = irreps_in
        self.irreps_out = o3.Irreps(irreps_out).simplify()

        values = dict()

        def evaluate(path):
            if path in values:
                return values[path]

            if isinstance(path, _INPUT):
                out = inputs[path.tensor]
                if (path.start, path.stop) != (0, self.irreps_in[path.tensor].dim):
                    out = out.narrow(-1, path.start, path.stop - path.start)
            if isinstance(path, _TP):
                x1 = evaluate(path.args[0]).node
                x2 = evaluate(path.args[1]).node
                out = fx.Proxy(graph.call_module(f"tp{tps.index(path.op)}", (x1, x2)), tracer)
            values[path] = out
            return out

        outs = []
        for vp_list in outputs:
            v, p = vp_list[0]
            out = evaluate(p)
            if abs(v - 1.0) > eps:
                out = v * out
            for v, p in vp_list[1:]:
                t = evaluate(p)
                if abs(v - 1.0) > eps:
                    t = v * t
                out = out + t
            outs.append(out)

        out = torch.cat(outs, dim=-1)
        graph.output(out.node)
        graphmod = fx.GraphModule(root, graph, "main")

        self._codegen_register({"main": graphmod})

    def __repr__(self):
        return (
            f"ReducedTensorProducts(\n"
            f"    in: {' times '.join(map(repr, self.irreps_in))}\n"
            f"    out: {self.irreps_out}\n"
            ")"
        )

    def forward(self, *xs):
        return self.main(*xs)
