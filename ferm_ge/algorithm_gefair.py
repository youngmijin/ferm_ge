import ctypes
import os
import pickle
import tempfile
import uuid
from functools import cached_property
from typing import Callable, Dict, List, Optional

import numpy as np
import numpy.ctypeslib as npct


class C_GEFAIR_RESULT(ctypes.Structure):
    pass


C_GEFAIR_RESULT._fields_ = [
    ("T", ctypes.c_size_t),
    ("D_bar", ctypes.POINTER(ctypes.c_size_t)),
    ("lambda_bar", ctypes.POINTER(ctypes.c_double)),
    ("hypothesis_history", ctypes.POINTER(ctypes.c_double)),
    ("I_alpha_history", ctypes.POINTER(ctypes.c_double)),
    ("err_history", ctypes.POINTER(ctypes.c_double)),
]


class GEFairResultSM:
    """
    A wrapper for the C++ GEFAIR_RESULT class. This class utilizes a shared
    memory to the C++ object using `numpy.ctypeslib.as_array`, so it can load
    result values quickly.
    """

    def __init__(
        self,
        c_result_p: ctypes.c_void_p,
        free_fn: Callable[[ctypes.c_void_p], None],
        includes_ge_history: bool,
        thr_candidates: List[float],
    ):
        self.c_result_p = c_result_p
        self.c_result = C_GEFAIR_RESULT.from_address(c_result_p)  # type: ignore

        self.free_fn = free_fn
        self.includes_ge_history = includes_ge_history
        self.thr_candidates = thr_candidates

    def __del__(self):
        self.free_fn(self.c_result_p)

    @property
    def T(self) -> int:
        return int(self.c_result.T)

    @cached_property
    def D_bar(self) -> Dict[float, int]:
        return {
            thr: self.c_result.D_bar[i]
            for i, thr in enumerate(self.thr_candidates)
        }

    @property
    def lambda_bar(self) -> np.ndarray:
        return npct.as_array(self.c_result.lambda_bar, (self.T,))

    @property
    def hypothesis_history(self) -> Optional[np.ndarray]:
        if self.includes_ge_history:
            return npct.as_array(self.c_result.hypothesis_history, (self.T,))
        else:
            return None

    @property
    def I_alpha_history(self) -> Optional[np.ndarray]:
        if self.includes_ge_history:
            return npct.as_array(self.c_result.I_alpha_history, (self.T,))
        else:
            return None

    @property
    def err_history(self) -> Optional[np.ndarray]:
        if self.includes_ge_history:
            return npct.as_array(self.c_result.err_history, (self.T,))
        else:
            return None

    def save_to(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)


class GEFairSolverC:
    cpp_path = os.path.join(
        os.path.dirname(__file__), "algorithm_gefair_impl.cpp"
    )

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
            ctypes.c_bool,
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
        collect_ge_history: bool = False,
    ) -> GEFairResultSM:
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
            ctypes.c_bool(collect_ge_history),
        )
        return GEFairResultSM(
            result_struct_p,
            self.lib.free_gefair_result,
            collect_ge_history,
            thr_candidates,
        )

    @staticmethod
    def compile_gefair() -> str:
        lib_path = os.path.join(
            tempfile.gettempdir(),
            f"algorithm_gefair_impl_{uuid.uuid4().hex}.so",
        )
        os.system(
            f"g++ -O3 -shared -std=c++11 -fPIC {GEFairSolverC.cpp_path} -o {lib_path}"
        )
        return lib_path
