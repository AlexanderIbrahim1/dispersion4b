import pytest

from contextlib import nullcontext
from dataclasses import dataclass

from dispersion4b.shortrange.short_range_functions import ExponentialDecay
from dispersion4b.shortrange.short_range_functions import ExponentialDecayOrder2


@pytest.fixture(scope="class")
def ed_args():
    @dataclass(frozen=True)
    class ExponentialDecayArguments:
        coeff: float
        expon: float

    return ExponentialDecayArguments(1.0, 1.0)


@pytest.mark.usefixtures("ed_args")
class TestExponentialDecay:
    def test_basic_functionality(self, ed_args):
        ed = ExponentialDecay(ed_args.coeff, ed_args.expon)

        assert ed(0.0) == pytest.approx(1.0)
        assert ed(3.0) < ed(2.0) < ed(1.0) < ed_args.coeff

    @pytest.mark.parametrize("bad_expon", [-1.0, 0.0])
    def test_raises_nonpositive_expon(self, bad_expon, ed_args):
        with pytest.raises(ValueError):
            ExponentialDecay(ed_args.coeff, bad_expon)


@pytest.fixture(scope="class")
def edo2_args():
    @dataclass(frozen=True)
    class ExponentialDecayOrder2Arguments:
        coeff: float
        expon_lin: float
        expon_sq: float

    return ExponentialDecayOrder2Arguments(1.0, 1.0, 1.0)


@pytest.mark.usefixtures("edo2_args")
class TestExponentialDecayOrder2:
    def test_basic_functionality(self, edo2_args):
        edo2 = ExponentialDecayOrder2(
            edo2_args.coeff, edo2_args.expon_lin, edo2_args.expon_sq
        )

        assert edo2(0.0) == pytest.approx(1.0)
        assert edo2(3.0) < edo2(2.0) < edo2(1.0) < edo2_args.coeff

    @pytest.mark.parametrize("bad_expon_sq", [-1.0, 0.0])
    def test_raises_nonpositive_expon_sq(self, bad_expon_sq, edo2_args):
        with pytest.raises(ValueError):
            ExponentialDecayOrder2(edo2_args.coeff, edo2_args.expon_lin, bad_expon_sq)

    @pytest.mark.parametrize("expon_lin", [-1.0, 0.0, 1.0])
    def test_all_expon_lin_allowed(self, expon_lin, edo2_args):
        with nullcontext():
            assert (
                ExponentialDecayOrder2(edo2_args.coeff, expon_lin, edo2_args.expon_sq)
                is not None
            )
