import ctypes
import os
import tempfile
import uuid
from functools import cached_property
from typing import Callable

import numpy as np
import numpy.ctypeslib as npct
from numpy.typing import NDArray


class C_GEFAIR_RESULT(ctypes.Structure):
    pass


C_GEFAIR_RESULT._fields_ = [
    ("T", ctypes.c_size_t),
    ("D_bar", ctypes.c_double),
    ("lambda_bar", ctypes.c_double),
    ("hypi_stat", ctypes.POINTER(ctypes.c_size_t)),
    ("hypi_t", ctypes.POINTER(ctypes.c_size_t)),
]


class GEFairResult:
    """A wrapper for the C++ GEFAIR_RESULT class (C_GEFAIR_RESULT)."""

    def __init__(
        self,
        thr_candidates: list[float],
        c_result_p: ctypes.c_void_p,
        free_fn: Callable[[ctypes.c_void_p], None],
    ):
        self.thr_candidates = thr_candidates

        self.c_result_p = c_result_p
        self.c_result = C_GEFAIR_RESULT.from_address(c_result_p)  # type: ignore

        self.free_fn = free_fn

    def __del__(self):
        self.free_fn(self.c_result_p)

    @property
    def T(self) -> int:
        return int(self.c_result.T)

    @property
    def D_bar(self) -> float:
        return float(self.c_result.D_bar)

    @property
    def lambda_bar(self) -> float:
        return float(self.c_result.lambda_bar)

    @cached_property
    def hyp_stat(self) -> dict[float, int]:
        return {
            thr: int(self.c_result.hypi_stat[i])
            for i, thr in enumerate(self.thr_candidates)
        }

    @property
    def hypi_t(self) -> NDArray[np.intp] | None:
        if self.c_result.hypi_t:
            return npct.as_array(
                self.c_result.hypi_t,
                (self.T,),
            ).astype(np.intp)
        else:
            return None


class GEFairSolver:
    cpp_path = os.path.join(os.path.dirname(__file__), "alg_gefair_impl.cpp")

    def __init__(self, lib_path: str):
        self.lib_path = lib_path
        self.lib = ctypes.cdll.LoadLibrary(lib_path)

        self.lib.solve_gefair.argtypes = [
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
            ctypes.c_double,
        ]
        self.lib.solve_gefair.restype = ctypes.c_void_p

        self.lib.free_gefair_result.argtypes = [ctypes.c_void_p]
        self.lib.free_gefair_result.restype = None

        self.flag_trace_hypi_t = ctypes.c_bool.in_dll(
            self.lib, "FLAG_TRACE_HYPI_T"
        ).value

        self.size_of_double = ctypes.c_size_t.in_dll(
            self.lib, "SIZE_OF_DOUBLE"
        ).value
        self.size_of_size_t = ctypes.c_size_t.in_dll(
            self.lib, "SIZE_OF_SIZE_T"
        ).value

    def solve_gefair(
        self,
        thr_candidates: list[float],
        I_alpha_cache: list[float],
        err_cache: list[float],
        lambda_max: float,
        nu: float,
        alpha: float,
        gamma: float,
        c: float,
        a: float,
    ) -> GEFairResult:
        result_struct_p = self.lib.solve_gefair(
            ctypes.c_size_t(len(thr_candidates)),
            (ctypes.c_double * len(thr_candidates))(*thr_candidates),
            (ctypes.c_double * len(I_alpha_cache))(*I_alpha_cache),
            (ctypes.c_double * len(err_cache))(*err_cache),
            ctypes.c_double(lambda_max),
            ctypes.c_double(nu),
            ctypes.c_double(alpha),
            ctypes.c_double(gamma),
            ctypes.c_double(c),
            ctypes.c_double(a),
        )
        return GEFairResult(
            thr_candidates,
            result_struct_p,
            self.lib.free_gefair_result,
        )

    def predict_memory_usage(
        self,
        thr_granularity: int,
        lambda_max: float,
        nu: float,
        alpha: float,
        gamma: float,
        c: float,
        a: float,
    ) -> float:
        """
        Predicts approximated memory consumption in bytes during algorithm.
        """

        mem_consumption: float = 0.0

        # memory for w0_mult and w1_mult
        mem_consumption += self.size_of_double * thr_granularity * 2

        # memory for hypi_stat
        mem_consumption += self.size_of_size_t * thr_granularity

        if self.flag_trace_hypi_t:
            ca: float = (c + a) / (c - a)
            if alpha == 0:
                I_up = np.log(ca)
            elif alpha == 1:
                I_up = ca * np.log(ca)
            else:
                I_up = (np.power(ca, alpha) - 1) / np.abs((alpha - 1) * alpha)
            A_alpha = 1 + lambda_max * (gamma + I_up)
            T = 4 * A_alpha * A_alpha * np.log(2) / (nu * nu)

            # memory for hypi_t
            mem_consumption += self.size_of_size_t * T

        return mem_consumption

    @staticmethod
    def compile_gefair(trace_hypi_t: bool = False) -> str:
        lib_path = os.path.join(
            tempfile.gettempdir(),
            f"alg_gefair_impl_{uuid.uuid4().hex}.so",
        )

        addi_arg_list = []
        if trace_hypi_t:
            addi_arg_list.append(f"-DTRACE_HYPI_T")

        addi_args = " ".join(addi_arg_list)
        os.system(
            f"g++ -O3 -shared -std=c++11 -fPIC {addi_args} {GEFairSolver.cpp_path} -o {lib_path}"
        )
        return lib_path