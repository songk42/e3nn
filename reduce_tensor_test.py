import tempfile

import torch
from e3nn import o3
from e3nn.o3 import ReducedTensorProducts
from e3nn.util.test import assert_auto_jitable, assert_equivariant


def isapprox(A, B, atol=1e-8, rtol=1e-5):
    close = torch.isclose(A, B, rtol, atol).flatten()
    for c in close:
        if not c: return False
    return True


def test_equivariance():
    irreps_in = "1o"
    tp = ReducedTensorProducts("ijkl=jikl=klij", i = irreps_in)
    Q = tp.change_of_basis

    abc = o3.rand_angles(1)

    R = o3.Irreps(irreps_in).D_from_angles(*abc)
    D = tp.irreps_out.D_from_angles(*abc)

    Q1 = torch.einsum('zijkl,mia,njb,okc,pld->zabcd', Q, R, R, R, R)
    Q2 = torch.einsum('xwz,zijkl->wijkl', D, Q)

    assert isapprox(Q1, Q2, atol=1e-5)


def test_permutation():
    tp = ReducedTensorProducts("ijkl=jikl=klij", i = "1o")
    Q = tp.change_of_basis

    Q1 = torch.einsum('zijkl->zjikl', Q)
    assert isapprox(Q1, Q, atol=1e-5)

    Q2 = torch.einsum('zijkl->zklji', Q)
    assert isapprox(Q2, Q, atol=1e-5)


def test_permutation2():
    tp = ReducedTensorProducts("ijk=-jik=jki", i = "1o")
    Q = tp.change_of_basis

    Q1 = torch.einsum('zijk->zjik', Q) * -1
    assert isapprox(Q1, Q, atol=1e-5)

    Q2 = torch.einsum('zijk->zjki', Q)
    assert isapprox(Q2, Q, atol=1e-5)


def test_orthogonality():
    tp = ReducedTensorProducts("ijkl=jikl=klij", i = "1o")
    Q = tp.change_of_basis
    n2 = torch.einsum('zijkl,wijkl->zw', Q, Q)
    assert isapprox(n2, torch.eye(Q.shape[0]), atol=1e-5)


def test_save_load():
    tp1 = ReducedTensorProducts("ij=-ji", i="5x0e + 1e")
    with tempfile.NamedTemporaryFile(suffix=".pth") as tmp:
        torch.save(tp1, tmp.name)
        tp2 = torch.load(tmp.name)

    xs = (torch.randn(2, 5 + 3), torch.randn(2, 5 + 3))
    assert torch.allclose(tp1(*xs), tp2(*xs))

    assert torch.allclose(tp1.change_of_basis, tp2.change_of_basis)


def test_antisymmetric_matrix(float_tolerance):
    tp = ReducedTensorProducts("ij=-ji", i="5x0e + 1e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(2, 5 + 3)
    assert (tp(*x) - torch.einsum("xij,i,j", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xij->xji", Q)).abs().max() < float_tolerance


def test_reduce_tensor_Levi_Civita_symbol(float_tolerance):
    tp = ReducedTensorProducts("ijk=-ikj=-jik", i="1e")
    assert tp.irreps_out == o3.Irreps("0e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(3, 3)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_antisymmetric_L2(float_tolerance):
    tp = ReducedTensorProducts("ijk=-ikj=-jik", i="2e")

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(3, 5)
    assert (tp(*x) - torch.einsum("xijk,i,j,k", Q, *x)).abs().max() < float_tolerance

    assert (Q + torch.einsum("xijk->xikj", Q)).abs().max() < float_tolerance
    assert (Q + torch.einsum("xijk->xjik", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor(float_tolerance):
    tp = ReducedTensorProducts("ijkl=jikl=klij", i="1e")
    assert tp.irreps_out.dim == 21

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


def test_reduce_tensor_elasticity_tensor_parity(float_tolerance):
    tp = ReducedTensorProducts("ijkl=jikl=klij", i="1o")
    assert tp.irreps_out.dim == 21
    assert all(ir.p == 1 for _, ir in tp.irreps_out)

    assert_equivariant(tp, irreps_in=tp.irreps_in, irreps_out=tp.irreps_out)
    assert_auto_jitable(tp)

    Q = tp.change_of_basis
    x = torch.randn(4, 3)
    assert (tp(*x) - torch.einsum("xijkl,i,j,k,l", Q, *x)).abs().max() < float_tolerance

    assert (Q - torch.einsum("xijkl->xjikl", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xijlk", Q)).abs().max() < float_tolerance
    assert (Q - torch.einsum("xijkl->xklij", Q)).abs().max() < float_tolerance


if __name__ == "__main__":
    # print("test_equivariance")
    # test_equivariance()
    # print("test_permutation")
    # test_permutation()
    # print("test_permutation2")
    # test_permutation2()
    # print("test_orthogonality")
    # test_orthogonality()
    # print("test_save_load")
    # test_save_load()
    # print("test_antisymmetric_matrix")
    # test_antisymmetric_matrix(1e-5)
    print("test_reduce_tensor_Levi_Civita_symbol")
    test_reduce_tensor_Levi_Civita_symbol(1e-5)
    print("test_reduce_tensor_antisymmetric_L2")
    test_reduce_tensor_antisymmetric_L2(1e-5)
    print("test_reduce_tensor_elasticity_tensor")
    test_reduce_tensor_elasticity_tensor(1e-5)
    print("test_reduce_tensor_elasticity_tensor_parity")
    test_reduce_tensor_elasticity_tensor_parity(1e-5)
