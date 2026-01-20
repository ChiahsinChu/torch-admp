# SPDX-License-Identifier: LGPL-3.0-or-later
import importlib
import re
import sys
import types
from typing import Callable, List
from unittest.mock import patch


def run_test_without_packages(
    func: Callable,
    pkg_names: List[str] | str,
    reload_module: types.ModuleType,
    **kwargs,
) -> None:
    """Decorator to run a test function with specified packages mocked as absent."""
    # Explicitly remove it if it happens to be installed in the real env
    if isinstance(pkg_names, str):
        # find all regex matches
        pkg_names = [
            pkg_name
            for pkg_name in sys.modules.keys()
            if re.search(pkg_names, pkg_name)
        ]
    try:
        with patch.dict(sys.modules, {pkg_name: None for pkg_name in pkg_names}):
            importlib.reload(reload_module)
            func(**kwargs)
    finally:
        importlib.reload(reload_module)
