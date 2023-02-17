import pytest

from dataclasses import dataclass

from dispersion4b.shortrange.attenuation import SilveraGoldmanAttenuation


@pytest.fixture(scope="class")
def sga_args():
    @dataclass
    class SilveraGoldmanAttenuationArguments:
        r_cutoff: float
        expon_coeff: float

    yield SilveraGoldmanAttenuationArguments(1.0, 1.0)


@pytest.mark.usefixtures("sga_args")
class TestSilveraGoldmanAttenuation:
    def test_basic_functionality(self):
        sg_atten = SilveraGoldmanAttenuation(1.0, 1.0)

        assert sg_atten(1.1) == pytest.approx(1.0)
        assert sg_atten(1.0) == pytest.approx(1.0)
        assert 0.0 < sg_atten(0.9) < 1.0

    @pytest.mark.parametrize("bad_r_cutoff", [-1.0, 0.0])
    def test_raises_nonpositive_r_cutoff(self, bad_r_cutoff, sga_args):
        with pytest.raises(ValueError):
            SilveraGoldmanAttenuation(bad_r_cutoff, sga_args.expon_coeff)

    @pytest.mark.parametrize("bad_expon_coeff", [-1.0, 0.0])
    def test_raises_nonpositive_expon_cutoff(self, bad_expon_coeff, sga_args):
        with pytest.raises(ValueError):
            SilveraGoldmanAttenuation(sga_args.r_cutoff, bad_expon_coeff)
