#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""

import os
import unittest
import io
import contextlib

from mili import grizinterface

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestGrizInterfaceRuns(unittest.TestCase):
    """Test that GrizInterface runs for various plot files."""

    def test_d3samp6(self):
        file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
        griz = grizinterface.open_griz_interface( file_name, experimental=True )
        self.assertIsInstance( griz, grizinterface.GrizInterface)

    def test_basic1(self):
        file_name = os.path.join(dir_path,'data','parallel','basic1','basic1.plt')
        griz = grizinterface.open_griz_interface( file_name, experimental=True )
        self.assertIsInstance( griz, grizinterface.GrizInterface)


class TestGrizInterfaceErrors(unittest.TestCase):
    """Test expected errors from GrizInterface."""
    def test_directory_does_not_exist(self):
        file_name = os.path.join(dir_path,'data','parallel','bad_dir','d3samp6.plt')
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            griz = grizinterface.open_griz_interface( file_name, experimental=True )
        self.assertTrue("Mili File Error: Cannot locate mili file director" in f.getvalue())
        self.assertEqual( griz, None )

    def test_database_does_not_exist(self):
        file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp4.plt')
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            griz = grizinterface.open_griz_interface( file_name, experimental=True )
        self.assertTrue("Mili File Error: No A-files for procs '' with base name" in f.getvalue())
        self.assertEqual( griz, None )
