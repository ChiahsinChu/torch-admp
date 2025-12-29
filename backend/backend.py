# SPDX-License-Identifier: LGPL-3.0-or-later
"""A PEP-517 backend for torch-admp."""

from scikit_build_core import build as _orig

__all__ = [
    "build_sdist",
    "build_wheel",
    "get_requires_for_build_sdist",
    "get_requires_for_build_wheel",
    "prepare_metadata_for_build_wheel",
]


def __dir__() -> list[str]:
    return __all__


prepare_metadata_for_build_wheel = _orig.prepare_metadata_for_build_wheel
build_wheel = _orig.build_wheel
build_sdist = _orig.build_sdist
get_requires_for_build_sdist = _orig.get_requires_for_build_sdist
prepare_metadata_for_build_editable = _orig.prepare_metadata_for_build_editable
build_editable = _orig.build_editable


def get_requires_for_build_wheel(
    config_settings: dict,
) -> list[str]:
    return _orig.get_requires_for_build_wheel(config_settings)


def get_requires_for_build_editable(
    config_settings: dict,
) -> list[str]:
    return _orig.get_requires_for_build_editable(config_settings)
