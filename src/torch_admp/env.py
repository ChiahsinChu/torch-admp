# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import ml_dtypes
import numpy as np
import torch

# FLOAT_PREC
dp_float_prec = os.environ.get("DP_INTERFACE_PREC", "high").lower()
if dp_float_prec in ("high", ""):
    # default is high
    GLOBAL_NP_FLOAT_PRECISION = np.float64
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "double"
elif dp_float_prec == "low":
    GLOBAL_NP_FLOAT_PRECISION = np.float32
    GLOBAL_ENER_FLOAT_PRECISION = np.float64
    global_float_prec = "float"
else:
    raise RuntimeError(
        f"Unsupported float precision option: {dp_float_prec}. Supported: high,"
        "low. Please set precision with environmental variable "
        "DP_INTERFACE_PREC."
    )

LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)
if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

torch.set_default_dtype(torch.float64)
torch.set_default_device(DEVICE)

PT_PRECISION_DICT = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bfloat16": torch.bfloat16,
    "bool": torch.bool,
}
NP_PRECISION_DICT = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "half": np.float16,
    "single": np.float32,
    "double": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
    "default": GLOBAL_NP_FLOAT_PRECISION,
    # NumPy doesn't have bfloat16 (and doesn't plan to add)
    # ml_dtypes is a solution, but it seems not supporting np.save/np.load
    # hdf5 hasn't supported bfloat16 as well (see https://forum.hdfgroup.org/t/11975)
    "bfloat16": ml_dtypes.bfloat16,
}
