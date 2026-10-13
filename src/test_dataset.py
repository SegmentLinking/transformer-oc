"""
test_dataset.py - tests for the eta/phi mask in dataset.py

You can run me like:
> cd src && apptainer exec /ceph/users/atuna/ml-tracking.sif python3 test_dataset.py
"""
import math
import torch

from dataset import eta_phi_mask, validate_mask_args


def test_eta_phi_mask_inside_outside():
    eta = torch.tensor([0.0, 0.4, -0.4, 0.6, 0.0, 0.0])
    phi = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.4, 0.6])
    mask = eta_phi_mask(eta, phi, 0.0, 0.0, 0.5, 0.5)
    assert mask.tolist() == [True, True, True, False, True, False]


def test_eta_phi_mask_off_center():
    eta = torch.tensor([1.0, 1.4, 0.4])
    phi = torch.tensor([-2.0, -2.2, -2.0])
    mask = eta_phi_mask(eta, phi, 1.2, -2.0, 0.5, 0.3)
    assert mask.tolist() == [True, True, False]


def test_eta_phi_mask_phi_wraparound():
    # center at phi = 3.0 with half width 0.5 covers [2.5, pi] and [-pi, -2.78...]
    eta = torch.zeros(4)
    phi = torch.tensor([3.0, -3.0, 2.4, -2.7])
    mask = eta_phi_mask(eta, phi, 0.0, 3.0, None, 0.5)
    assert mask.tolist() == [True, True, False, False]


def test_eta_phi_mask_none_keeps_all():
    eta = torch.tensor([-2.5, 0.0, 2.5])
    phi = torch.tensor([-3.1, 0.0, 3.1])
    assert eta_phi_mask(eta, phi, 0.0, 0.0, None, None).all()


def test_eta_phi_mask_eta_boundary_inclusive():
    # phi boundaries are also inclusive, but are not tested exactly because
    # the wrapped dphi from atan2 is subject to float rounding
    eta = torch.tensor([-0.5, 0.5])
    phi = torch.zeros(2)
    assert eta_phi_mask(eta, phi, 0.0, 0.0, 0.5, None).all()


def test_eta_phi_mask_large_half_width_keeps_all():
    eta = torch.tensor([-2.6, 0.0, 2.6])
    phi = torch.tensor([-math.pi, 0.0, math.pi])
    assert eta_phi_mask(eta, phi, 2.6, -math.pi, 5.2, math.pi).all()


def assert_raises_value_error(func, **kwargs):
    try:
        func(**kwargs)
    except ValueError:
        return
    raise AssertionError(f"{func.__name__}({kwargs}) did not raise ValueError")


def test_validate_mask_args_full_event():
    validate_mask_args(eta_half_width=None, eta_center=0.0, phi_half_width=None, phi_center=0.0)


def test_validate_mask_args_random_center():
    validate_mask_args(eta_half_width=0.5, eta_center=None, phi_half_width=0.5, phi_center=None)


def test_validate_mask_args_fixed_center():
    validate_mask_args(eta_half_width=0.5, eta_center=0.0, phi_half_width=0.5, phi_center=1.0)


def test_validate_mask_args_no_half_width_no_center_raises():
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=None,
        eta_center=None,
        phi_half_width=0.5,
        phi_center=None,
    )
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=0.5,
        eta_center=None,
        phi_half_width=None,
        phi_center=None,
    )


def test_validate_mask_args_no_half_width_nonzero_center_raises():
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=None,
        eta_center=1.0,
        phi_half_width=None,
        phi_center=0.0,
    )
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=None,
        eta_center=0.0,
        phi_half_width=None,
        phi_center=1.0,
    )


def test_validate_mask_args_nonpositive_half_width_raises():
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=0.0,
        eta_center=None,
        phi_half_width=0.5,
        phi_center=None,
    )
    assert_raises_value_error(
        validate_mask_args,
        eta_half_width=0.5,
        eta_center=None,
        phi_half_width=-0.5,
        phi_center=None,
    )


if __name__ == "__main__":
    for name, func in list(globals().items()):
        if name.startswith("test_"):
            func()
            print(f"{name}: passed")
