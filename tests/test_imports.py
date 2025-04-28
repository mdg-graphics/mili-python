#!/usr/bin/env python3
"""
Testing import Mili modules.

SPDX-License-Identifier: (MIT)
"""

import unittest

class TestImports(unittest.TestCase):
    """Test importing the various Mili modules."""

    def test_import_modules(self):
        """Test to try to catch import errors."""
        from mili import adjacency
        from mili import afileIO
        from mili import append_states
        from mili import datatypes
        from mili import derived
        from mili import geometric_mesh_info
        # Skip grizinterface.py
        from mili import milidatabase
        from mili import miliinternal
        # Skip parallel.py
        from mili import plotting
        from mili import reader
        from mili import reductions
        from mili import utils