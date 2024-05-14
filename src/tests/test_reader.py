#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili.reader import open_database
from mili.milidatabase import MiliDatabase
from mili.afileIO import MiliFileNotFoundError
from mili.miliinternal import _MiliInternal
from mili.parallel import LoopWrapper, ServerWrapper

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestOpenDatabase(unittest.TestCase):

    #==============================================================================
    def test_open_serial_sstate(self):
        file_path = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
        db = open_database(file_path)
        self.assertTrue(isinstance(db, MiliDatabase))
        self.assertTrue(isinstance(db._mili, _MiliInternal))

    #==============================================================================
    def test_open_serial_mstate(self):
        file_path = os.path.join(dir_path,'data','serial','mstate','d3samp6.plt_c')
        db = open_database(file_path)
        self.assertTrue(isinstance(db, MiliDatabase))
        self.assertTrue(isinstance(db._mili, _MiliInternal))

    #==============================================================================
    def test_open_parallel_loopwrapper(self):
        file_path = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
        db = open_database(file_path, suppress_parallel=True)
        self.assertTrue(isinstance(db, MiliDatabase))
        self.assertTrue(isinstance(db._mili, LoopWrapper))

    #==============================================================================
    def test_open_parallel_serverwrapper(self):
        file_path = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
        db = open_database(file_path, suppress_parallel=False)
        self.assertTrue(isinstance(db, MiliDatabase))
        self.assertTrue(isinstance(db._mili, ServerWrapper))
        db.close()

    #==============================================================================
    def test_open_processor_subset(self):
        file_path = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
        db = open_database(file_path, suppress_parallel=False, procs=[0])
        self.assertTrue(isinstance(db, MiliDatabase))
        self.assertTrue(isinstance(db._mili, _MiliInternal))

    #==============================================================================
    def test_open_directory_does_not_exist(self):
        with self.assertRaises(MiliFileNotFoundError):
            db = open_database("does_not_exist/d3samp6.plt")

    #==============================================================================
    def test_open_database_does_not_exist(self):
        file_path = os.path.join(dir_path,'data','serial','sstate','does-not-exist.plt')
        with self.assertRaises(MiliFileNotFoundError):
            db = open_database(file_path)


if __name__ == "__main__":
    unittest.main()
