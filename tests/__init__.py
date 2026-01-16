# SPDX-License-Identifier: LGPL-3.0-or-later
import os

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# testing purposes; device should always be set explicitly
torch.set_default_device("cuda:9999999")

# Make sure DDP uses correct device if applicable
LOCAL_RANK = os.environ.get("LOCAL_RANK")
LOCAL_RANK = int(0 if LOCAL_RANK is None else LOCAL_RANK)

if os.environ.get("DEVICE") == "cpu" or torch.cuda.is_available() is False:
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")

SEED = 1024
