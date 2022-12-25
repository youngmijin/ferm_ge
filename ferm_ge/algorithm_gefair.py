import ctypes
import os
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class GEFairResult:
    T: int
    D_bar: Dict[float, int]
    lambda_bar: List[float]
    hypothesis_history: List[float]


class GEFAIR_RESULT(ctypes.Structure):
    pass


GEFAIR_RESULT._fields_ = [
    ("T", ctypes.c_size_t),
    ("D_bar", ctypes.POINTER(ctypes.c_size_t)),
    ("lambda_bar", ctypes.POINTER(ctypes.c_double)),
    ("hypothesis_history", ctypes.POINTER(ctypes.c_double)),
]


class GEFairSolver:
    cpp_path = os.path.join(
        os.path.dirname(__file__), "algorithm_gefair_impl.cpp"
    )
    lib_path = os.path.join(
        os.path.dirname(__file__), "algorithm_gefair_impl.so"
    )

    def __init__(self):
        self.lib = ctypes.cdll.LoadLibrary(self.lib_path)

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
        ]
        self.lib.solve_gefair.restype = ctypes.c_void_p

        self.lib.free_gefair_result.argtypes = [ctypes.c_void_p]
        self.lib.free_gefair_result.restype = None

    def solve_gefair(
        self,
        thr_candidates: List[float],
        I_alpha_cache: List[float],
        err_cache: List[float],
        alpha: float,
        lambda_max: float,
        nu: float,
        r: float,
        gamma: float,
    ) -> GEFairResult:
        result_struct_p = self.lib.solve_gefair(
            ctypes.c_size_t(len(thr_candidates)),
            (ctypes.c_double * len(thr_candidates))(*thr_candidates),
            (ctypes.c_double * len(I_alpha_cache))(*I_alpha_cache),
            (ctypes.c_double * len(err_cache))(*err_cache),
            ctypes.c_double(alpha),
            ctypes.c_double(lambda_max),
            ctypes.c_double(nu),
            ctypes.c_double(r),
            ctypes.c_double(gamma),
        )
        result_struct = GEFAIR_RESULT.from_address(result_struct_p)
        result = GEFairResult(
            T=result_struct.T,
            D_bar={
                t: result_struct.D_bar[i]
                for i, t in enumerate(thr_candidates)
                if result_struct.D_bar[i] != 0
            },
            lambda_bar=[
                result_struct.lambda_bar[i] for i in range(result_struct.T)
            ],
            hypothesis_history=[
                result_struct.hypothesis_history[i]
                for i in range(result_struct.T)
            ],
        )
        self.lib.free_gefair_result(result_struct_p)
        return result

    @staticmethod
    def compile_gefair():
        os.system(
            f"g++ -O3 -shared -std=c++17 -fPIC {GEFairSolver.cpp_path} -o {GEFairSolver.lib_path}"
        )
