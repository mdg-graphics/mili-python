#!/usr/bin/env python3

"""
Copyright (c) 2016-present, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by:

 William Tobin (tobin6@llnl.gov),
 Ryan Hathaway (hathaway6@llnl.gov),
 Kevin Durrenberger (durrenberger1@llnl.gov).

 CODE-OCEC-16-056.
 All rights reserved.

 This file is part of Mili. For details, see TODO: <URL describing code
 and how to download source>.

 Please also read this link-- Our Notice and GNU Lesser General
 Public License.

 This program is free software; you can redistribute it and/or modify
 it under the terms of the GNU General Public License (as published by
 the Free Software Foundation) version 2.1 dated February 1999.

 This program is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms
 and conditions of the GNU General Public License for more details.

 You should have received a copy of the GNU Lesser General Public License
 along with this program; if not, write to the Free Software Foundation,
 Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

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