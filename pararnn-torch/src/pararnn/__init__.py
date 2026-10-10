"""Homework fork of pararnn-torch: ParaGRU / ParaLSTM only."""

from importlib.metadata import PackageNotFoundError, version

from pararnn.cells import ParaGRU, ParaLSTM, RNNCell
from pararnn.kernels import can_decode_step, decode_step, decode_wx
from pararnn.layers import ParaRNN
from pararnn.solvers import (
    NewtonConfig,
    NewtonDivergenceError,
    NewtonStats,
    compile_safe_config,
    newton_apply,
    sequential_apply,
)
from pararnn.verify import (
    AgreementError,
    AgreementReport,
    default_agreement_atol,
    verify_agreement,
)

__all__ = [
    "AgreementError",
    "AgreementReport",
    "NewtonConfig",
    "NewtonDivergenceError",
    "NewtonStats",
    "ParaGRU",
    "ParaLSTM",
    "ParaRNN",
    "RNNCell",
    "can_decode_step",
    "compile_safe_config",
    "decode_step",
    "decode_wx",
    "default_agreement_atol",
    "newton_apply",
    "sequential_apply",
    "verify_agreement",
]

try:
    __version__ = version("pararnn-torch")
except PackageNotFoundError:
    __version__ = "0.21.0+hw2"
