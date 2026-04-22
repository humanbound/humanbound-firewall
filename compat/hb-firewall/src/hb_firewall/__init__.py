# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Humanbound
"""DEPRECATED: hb-firewall has been renamed to humanbound-firewall.

This stub re-exports everything from humanbound_firewall and emits a
DeprecationWarning on first import. It will be yanked from PyPI on or
after 2026-06-20.
"""

import warnings

warnings.warn(
    "hb-firewall has been renamed to humanbound-firewall. "
    "Please update your dependencies: `pip install humanbound-firewall` "
    "and import from `humanbound_firewall` instead. "
    "This stub will be yanked from PyPI on or after 2026-06-20.",
    DeprecationWarning,
    stacklevel=2,
)

from humanbound_firewall import *  # noqa: F401, F403
from humanbound_firewall import __version__  # noqa: F401
