#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import os
import shutil
import unittest
from mili import reader
from mili.milidatabase import MiliPythonError
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestAppendStateSerial(unittest.TestCase):
    data_path = os.path.join(dir_path,'data','serial','sstate')
    required_plot_files = ["d3samp6.pltA", "d3samp6.plt00"]
    base_name = 'd3samp6.plt'

    def copy_required_plot_files(self, path, required_files, test_prefix):
        for file in required_files:
            fname = os.path.join(path, file)
            shutil.copyfile(fname, f"./{test_prefix}_{file}")

    def setUp(self):
        self.copy_required_plot_files(TestAppendStateSerial.data_path,
                                      TestAppendStateSerial.required_plot_files,
                                      self._testMethodName)

    def tearDown(self):
        # Only delete temp files if test passes.
        if all([err[1] == None for err in self._outcome.errors]):
            for f in TestAppendStateSerial.required_plot_files:
                os.remove(f"./{self._testMethodName}_{f}")

    #==============================================================================
    def test_serial_append_single_state_zero_out_false(self):
        """Test appending a single state to an existing database. zero_out=False"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=False )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 102)
        final_smap = smaps[-1]
        self.assertEqual(final_smap.file_number, 0)
        self.assertEqual(final_smap.file_offset, 1763460)
        self.assertEqual(final_smap.time, 100.00)
        self.assertEqual(final_smap.state_map_id, 0)

        # Test that we can query this new state.
        # State 102 have the same data as 101 so we can just compare those
        # to verify the write was correct.
        nodpos = mili.query("nodpos", "node", states=[101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        stress = mili.query("stress", "brick", states=[101,102])
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        stress = mili.query("stress", "beam", states=[101,102])
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 47)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

    #==============================================================================
    def test_serial_append_multiple_states_zero_out_false(self):
        """Test appending multiple states in a row to an existing database. zero_out is False"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=False )
        mili.append_state( 101.00, zero_out=False )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 103)

        smap_102 = smaps[-2]
        self.assertEqual(smap_102.file_number, 0)
        self.assertEqual(smap_102.file_offset, 1763460)
        self.assertEqual(smap_102.time, 100.00)
        self.assertEqual(smap_102.state_map_id, 0)

        smap_103 = smaps[-1]
        self.assertEqual(smap_103.file_number, 0)
        self.assertEqual(smap_103.file_offset, 1780920)
        self.assertEqual(smap_103.time, 101.00)
        self.assertEqual(smap_103.state_map_id, 0)

        # Test that we can query this new state.
        # States 102 and 103 should have the same data as 101 so we can just compare those
        # to verify the write was correct.
        nodpos = mili.query("nodpos", "node", states=[101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        state_103_data = nodpos['nodpos']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        stress = mili.query("stress", "brick", states=[101,102,103])
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        state_103_data = stress['stress']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        stress = mili.query("stress", "beam", states=[101,102,103])
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 47)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        state_103_data = stress['stress']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

    #==============================================================================
    def test_serial_append_single_state_zero_out_true(self):
        """Test appending a single state to an existing database. zero_out=True"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=True )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 102)
        final_smap = smaps[-1]
        self.assertEqual(final_smap.file_number, 0)
        self.assertEqual(final_smap.file_offset, 1763460)
        self.assertEqual(final_smap.time, 100.00)
        self.assertEqual(final_smap.state_map_id, 0)

        # Test that we can query this new state.
        # State 102 will be zeroed out except for nodpos and sand flags
        nodpos = mili.query("nodpos", "node", states=[101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        sand = mili.query("sand", "shell", states=[102])
        np.testing.assert_equal( sand['sand']['layout']['states'], [102])
        np.testing.assert_equal( sorted(sand['sand']['layout']['labels']), list(range(1, 13)) )
        state_102_data = sand['sand']['data'][0]
        np.testing.assert_equal( state_102_data, 1.0 )

        stress = mili.query("stress", "brick", states=[102])
        np.testing.assert_equal( stress['stress']['layout']['states'], [102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_102_data = stress['stress']['data'][0]
        np.testing.assert_equal( state_102_data, 0.0 )

    #==============================================================================
    def test_serial_append_multiple_states_zero_out_true(self):
        """Test appending multiple states in a row to an existing database. zero_out is True"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=True )
        mili.append_state( 101.00, zero_out=True )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 103)

        smap_102 = smaps[-2]
        self.assertEqual(smap_102.file_number, 0)
        self.assertEqual(smap_102.file_offset, 1763460)
        self.assertEqual(smap_102.time, 100.00)
        self.assertEqual(smap_102.state_map_id, 0)

        smap_103 = smaps[-1]
        self.assertEqual(smap_103.file_number, 0)
        self.assertEqual(smap_103.file_offset, 1780920)
        self.assertEqual(smap_103.time, 101.00)
        self.assertEqual(smap_103.state_map_id, 0)

        # Test that we can query this new state.
        # States 102 and 103 should be zeroed out except for nodpos and sand
        nodpos = mili.query("nodpos", "node", states=[101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        state_103_data = nodpos['nodpos']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        sand = mili.query("sand", "shell", states=[102,103])
        np.testing.assert_equal( sand['sand']['layout']['states'], [102,103])
        np.testing.assert_equal( sorted(sand['sand']['layout']['labels']), list(range(1, 13)) )
        state_102_data = sand['sand']['data'][0]
        state_103_data = sand['sand']['data'][1]
        np.testing.assert_equal( state_102_data, 1.0 )
        np.testing.assert_equal( state_103_data, 1.0 )

        stress = mili.query("stress", "brick", states=[102,103])
        np.testing.assert_equal( stress['stress']['layout']['states'], [102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_102_data = stress['stress']['data'][0]
        state_103_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_102_data, 0.0 )
        np.testing.assert_equal( state_103_data, 0.0 )

    #==============================================================================
    def test_serial_append_limit_states(self):
        """Test limiting state count per state file"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=False, limit_states_per_file=1 )
        mili.append_state( 101.00, zero_out=False, limit_states_per_file=1 )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 103)
        self.assertEqual(smaps[-1].file_number, 2)
        self.assertEqual(smaps[-2].file_number, 1)
        self.assertEqual(smaps[-3].file_number, 0)
        os.remove(f"{db_name}01")
        os.remove(f"{db_name}02")

    #==============================================================================
    def test_serial_append_limit_bytes(self):
        """Test limiting bytes per state file"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        mili.append_state( 100.00, zero_out=False, limit_bytes_per_file=1000 )
        mili.append_state( 101.00, zero_out=False, limit_bytes_per_file=1000 )

        # Test that state maps look correct.
        smaps = mili.state_maps()
        n_states = len(smaps)
        self.assertEqual(n_states, 103)
        self.assertEqual(smaps[-1].file_number, 2)
        self.assertEqual(smaps[-2].file_number, 1)
        self.assertEqual(smaps[-3].file_number, 0)
        os.remove(f"{db_name}01")
        os.remove(f"{db_name}02")

    #==============================================================================
    def test_serial_append_bad_time(self):
        """Test appending an invalid state time."""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name, merge_results=False )
        with self.assertRaises(MiliPythonError):
            mili.append_state( 0.00 )

class TestAppendStateParallel(unittest.TestCase):
    data_path = os.path.join(dir_path,'data','parallel','d3samp6')
    required_plot_files = ["d3samp6.plt000A", "d3samp6.plt00000",
                           "d3samp6.plt001A", "d3samp6.plt00100",
                           "d3samp6.plt002A", "d3samp6.plt00200",
                           "d3samp6.plt003A", "d3samp6.plt00300",
                           "d3samp6.plt004A", "d3samp6.plt00400",
                           "d3samp6.plt005A", "d3samp6.plt00500",
                           "d3samp6.plt006A", "d3samp6.plt00600",
                           "d3samp6.plt007A", "d3samp6.plt00700"]
    base_name = 'd3samp6.plt'

    def copy_required_plot_files(self, path, required_files, test_prefix):
        for file in required_files:
            fname = os.path.join(path, file)
            shutil.copyfile(fname, f"./{test_prefix}_{file}")

    def setUp(self):
        self.copy_required_plot_files(TestAppendStateParallel.data_path,
                                      TestAppendStateParallel.required_plot_files,
                                      self._testMethodName)


    def tearDown(self):
        # Only delete temp files if test passes.
        if all([err[1] == None for err in self._outcome.errors]):
            for f in TestAppendStateParallel.required_plot_files:
                os.remove(f"./{self._testMethodName}_{f}")

    #==============================================================================
    def test_parallel_append_single_state_zero_out_false(self):
        """Test appending a single state to an existing database. zero_out=False"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, merge_results=False )
        mili.append_state( 100.00, zero_out=False )

        # Test that state maps look correct.
        state_maps = mili.state_maps()
        self.assertEqual( len(state_maps), 8 )

        EXPECTED_OFFSETS = [243208, 225432, 208464, 294516, 294516, 212100, 287244, 287244]
        for idx,smaps in enumerate(state_maps):
            self.assertEqual(len(smaps), 102)
            self.assertEqual(smaps[-1].file_number, 0)
            self.assertEqual(smaps[-1].file_offset, EXPECTED_OFFSETS[idx])
            self.assertEqual(smaps[-1].time, 100.00)
            self.assertEqual(smaps[-1].state_map_id, 0)

        # Test that we can query this new state.
        # State 102 should have the same data as 101 so we can just compare those
        # to verify the write was correct.
        nodpos = reader.combine(mili.query("nodpos", "node", states=[101,102]))
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        stress = reader.combine(mili.query("stress", "brick", states=[101,102]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        stress = reader.combine(mili.query("stress", "beam", states=[101,102]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 47)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )
        mili.close()

    #==============================================================================
    def test_parallel_append_multiple_states_zero_out_false(self):
        """Test appending multiple states in a row to an existing database. zero_out is False"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, merge_results=False )
        mili.append_state( 100.00, zero_out=False )
        mili.append_state( 101.00, zero_out=False )

        # Test that state maps look correct.
        state_maps = mili.state_maps()
        self.assertEqual( len(state_maps), 8 )

        EXPECTED_OFFSETS_1 = [243208, 225432, 208464, 294516, 294516, 212100, 287244, 287244]
        EXPECTED_OFFSETS_2 = [245616, 227664, 210528, 297432, 297432, 214200, 290088, 290088]
        for idx,smaps in enumerate(state_maps):
            self.assertEqual(len(smaps), 103)

            self.assertEqual(smaps[-2].file_number, 0)
            self.assertEqual(smaps[-2].file_offset, EXPECTED_OFFSETS_1[idx])
            self.assertEqual(smaps[-2].time, 100.00)
            self.assertEqual(smaps[-2].state_map_id, 0)

            self.assertEqual(smaps[-1].file_number, 0)
            self.assertEqual(smaps[-1].file_offset, EXPECTED_OFFSETS_2[idx])
            self.assertEqual(smaps[-1].time, 101.00)
            self.assertEqual(smaps[-1].state_map_id, 0)

        # Test that we can query this new state.
        # States 102 and 103 should have the same data as 101 so we can just compare those
        # to verify the write was correct.
        nodpos = reader.combine(mili.query("nodpos", "node", states=[101,102,103]))
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], list(range(1, 145)) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        state_103_data = nodpos['nodpos']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        stress = reader.combine(mili.query("stress", "brick", states=[101,102,103]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        state_103_data = stress['stress']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        stress = reader.combine(mili.query("stress", "beam", states=[101,102,103]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [101,102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 47)) )
        state_101_data = stress['stress']['data'][0]
        state_102_data = stress['stress']['data'][1]
        state_103_data = stress['stress']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )
        mili.close()

    #==============================================================================
    def test_parallel_append_single_state_zero_out_true(self):
        """Test appending a single state to an existing database. zero_out=True"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False,  merge_results=False )
        mili.append_state( 100.00, zero_out=True )

        # Test that state maps look correct.
        state_maps = mili.state_maps()
        self.assertEqual( len(state_maps), 8 )

        EXPECTED_OFFSETS = [243208, 225432, 208464, 294516, 294516, 212100, 287244, 287244]
        for idx,smaps in enumerate(state_maps):
            self.assertEqual(len(smaps), 102)
            self.assertEqual(smaps[-1].file_number, 0)
            self.assertEqual(smaps[-1].file_offset, EXPECTED_OFFSETS[idx])
            self.assertEqual(smaps[-1].time, 100.00)
            self.assertEqual(smaps[-1].state_map_id, 0)

        # Test that we can query this new state.
        # State 102 will be zeroed out except for nodpos and sand flags
        nodpos = reader.combine(mili.query("nodpos", "node", states=[101,102]))
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        np.testing.assert_equal( state_101_data, state_102_data )

        sand = reader.combine(mili.query("sand", "shell", states=[102]))
        np.testing.assert_equal( sand['sand']['layout']['states'], [102])
        np.testing.assert_equal( sorted(sand['sand']['layout']['labels']), list(range(1, 13)) )
        state_102_data = sand['sand']['data'][0]
        np.testing.assert_equal( state_102_data, 1.0 )

        stress = reader.combine(mili.query("stress", "brick", states=[102]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [102])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_102_data = stress['stress']['data'][0]
        np.testing.assert_equal( state_102_data, 0.0 )
        mili.close()

    #==============================================================================
    def test_parallel_append_multiple_states_zero_out_true(self):
        """Test appending multiple states in a row to an existing database. zero_out is True"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, merge_results=False )
        mili.append_state( 100.00, zero_out=True )
        mili.append_state( 101.00, zero_out=True )

        # Test that state maps look correct.
        state_maps = mili.state_maps()
        self.assertEqual( len(state_maps), 8 )

        EXPECTED_OFFSETS_1 = [243208, 225432, 208464, 294516, 294516, 212100, 287244, 287244]
        EXPECTED_OFFSETS_2 = [245616, 227664, 210528, 297432, 297432, 214200, 290088, 290088]
        for idx,smaps in enumerate(state_maps):
            self.assertEqual(len(smaps), 103)

            self.assertEqual(smaps[-2].file_number, 0)
            self.assertEqual(smaps[-2].file_offset, EXPECTED_OFFSETS_1[idx])
            self.assertEqual(smaps[-2].time, 100.00)
            self.assertEqual(smaps[-2].state_map_id, 0)

            self.assertEqual(smaps[-1].file_number, 0)
            self.assertEqual(smaps[-1].file_offset, EXPECTED_OFFSETS_2[idx])
            self.assertEqual(smaps[-1].time, 101.00)
            self.assertEqual(smaps[-1].state_map_id, 0)

        # Test that we can query this new state.
        # States 102 and 103 should be zeroed out except for nodpos and sand
        nodpos = reader.combine(mili.query("nodpos", "node", states=[101,102,103]))
        np.testing.assert_equal( nodpos['nodpos']['layout']['states'], [101,102,103])
        np.testing.assert_equal( nodpos['nodpos']['layout']['labels'], range(1, 145) )
        state_101_data = nodpos['nodpos']['data'][0]
        state_102_data = nodpos['nodpos']['data'][1]
        state_103_data = nodpos['nodpos']['data'][2]
        np.testing.assert_equal( state_101_data, state_102_data )
        np.testing.assert_equal( state_101_data, state_103_data )

        sand = reader.combine(mili.query("sand", "shell", states=[102,103]))
        np.testing.assert_equal( sand['sand']['layout']['states'], [102,103])
        np.testing.assert_equal( sorted(sand['sand']['layout']['labels']), list(range(1, 13)) )
        state_102_data = sand['sand']['data'][0]
        state_103_data = sand['sand']['data'][1]
        np.testing.assert_equal( state_102_data, 1.0 )
        np.testing.assert_equal( state_103_data, 1.0 )

        stress = reader.combine(mili.query("stress", "brick", states=[102,103]))
        np.testing.assert_equal( stress['stress']['layout']['states'], [102,103])
        np.testing.assert_equal( sorted(stress['stress']['layout']['labels']), list(range(1, 37)) )
        state_102_data = stress['stress']['data'][0]
        state_103_data = stress['stress']['data'][1]
        np.testing.assert_equal( state_102_data, 0.0 )
        np.testing.assert_equal( state_103_data, 0.0 )
        mili.close()

    #==============================================================================
    def test_parallel_append_bad_time(self):
        """Test appending an invalid state time."""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, merge_results=False )
        with self.assertRaises(MiliPythonError):
            mili.append_state( 0.00 )
        mili.close()

class TestCopyNonStateDataSerial(unittest.TestCase):
    data_path = os.path.join(dir_path,'data','serial','sstate')
    base_name = 'd3samp6.plt'
    new_base_name = 'copy_d3samp6.plt'

    def dictionary_diff_helper(self, a, b):
        a_keys = a.keys()
        b_keys = b.keys()
        self.assertEqual(a_keys, b_keys)
        for key in a_keys:
            np.testing.assert_equal( a[key], b[key] )

    def setUp(self):
        self.db = reader.open_database( os.path.join(TestCopyNonStateDataSerial.data_path,
                                        TestCopyNonStateDataSerial.base_name), suppress_parallel=True,
                                        merge_results=False )

    def tearDown(self):
        if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}A"):
            os.remove(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}A")
        if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}00"):
            os.remove(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}00")

    #==============================================================================
    def test_copy(self):
        """Tests creating a copy of an existing database without the state data."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name, suppress_parallel=True, merge_results=False )

        # Check that state maps, times and state counts are all 0
        self.assertEqual( len(new_db.state_maps()), 0 )
        self.assertEqual( len(new_db.times()), 0 )
        self.assertEqual( new_db._mili.parameter('state_count'), 0 )

        # Check that everything else looks the same
        np.testing.assert_equal( new_db.nodes(), self.db.nodes() )
        self.dictionary_diff_helper( new_db.labels(), self.db.labels() )
        self.dictionary_diff_helper( new_db.connectivity(), self.db.connectivity() )
        self.assertEqual( new_db.mesh_dimensions(), self.db.mesh_dimensions() )
        self.assertEqual( new_db.element_sets(), self.db.element_sets() )
        self.dictionary_diff_helper(self.db._mili.state_variables(), new_db._mili.state_variables())
        self.assertEqual(self.db._mili.subrecords(), new_db._mili.subrecords())

    #==============================================================================
    def test_copy_append_state(self):
        """Tests copying existing database and then appending a state."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name, suppress_parallel=True, merge_results=False )
        new_db.append_state( 0.0 )

        # Check nodal positions
        initial_nodal_positions = self.db.nodes()
        new_db_nodpos = new_db.query("nodpos", "node", states=[1])
        np.testing.assert_equal( initial_nodal_positions, new_db_nodpos['nodpos']['data'][0] )

        # Check sand flags
        sand_classes = self.db.classes_of_state_variable("sand")
        for class_name in sand_classes:
            sand_flags = new_db.query("sand", class_name, states=[1])
            np.testing.assert_equal( sand_flags['sand']['data'][0], 1.0)

class TestCopyNonStateDataParallel(unittest.TestCase):
    data_path = os.path.join(dir_path,'data','parallel','d3samp6')
    base_name = 'd3samp6.plt'
    new_base_name = 'copy_d3samp6.plt'

    def dictionary_diff_helper(self, a, b):
        a_keys = a.keys()
        b_keys = b.keys()
        self.assertEqual(a_keys, b_keys)
        for key in a_keys:
            np.testing.assert_equal( a[key], b[key] )

    def setUp(self):
        self.db = reader.open_database( os.path.join(TestCopyNonStateDataParallel.data_path,
                                        TestCopyNonStateDataParallel.base_name),
                                        suppress_parallel=False,
                                        merge_results=False)

    def tearDown(self):
        self.db.close()
        for proc in ["000", "001", "002", "003", "004", "005", "006", "007"]:
            if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}A"):
                os.remove(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}A")
            if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}00"):
                os.remove(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}00")
            if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}01"):
                os.remove(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}01")

    #==============================================================================
    def test_copy(self):
        """Tests creating a copy of an existing database without the state data."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, merge_results=False )

        # Check that state maps, times and state counts are all 0 for all processors
        state_maps = new_db.state_maps()
        state_times = new_db.times()
        self.assertEqual(len(state_maps), 8)
        for processor_smap, processor_times in zip(state_maps, state_times):
            self.assertEqual( len(processor_smap), 0 )
            self.assertEqual( len(processor_times), 0 )
        self.assertEqual( new_db._mili.parameter('state_count'), [0,0,0,0,0,0,0,0] )

        # Check that everything else looks the same
        np.testing.assert_equal( new_db.nodes(), self.db.nodes() )
        for orig_labels, new_labels in zip(self.db.labels(), new_db.labels() ):
            self.dictionary_diff_helper( orig_labels, new_labels )
        for orig_conns, new_conns in zip(self.db.connectivity(), new_db.connectivity() ):
            self.dictionary_diff_helper( orig_conns, new_conns )
        self.assertEqual( new_db.mesh_dimensions(), self.db.mesh_dimensions() )
        self.assertEqual( new_db.element_sets(), self.db.element_sets() )
        self.assertEqual( new_db._mili.state_variables(), self.db._mili.state_variables() )
        self.assertEqual( new_db._mili.subrecords(), self.db._mili.subrecords() )
        new_db.close()

    #==============================================================================
    def test_copy_append_state(self):
        """Tests copying existing database and then appending a state."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, merge_results=False )
        new_db.append_state( 0.0 )

        # Check nodal positions
        initial_nodal_positions = self.db.nodes()
        new_db_nodpos = new_db.query("nodpos", "node", states=[1])
        np.testing.assert_equal( initial_nodal_positions[0], new_db_nodpos[0]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[1], new_db_nodpos[1]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[2], new_db_nodpos[2]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[3], new_db_nodpos[3]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[4], new_db_nodpos[4]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[5], new_db_nodpos[5]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[6], new_db_nodpos[6]['nodpos']['data'][0] )
        np.testing.assert_equal( initial_nodal_positions[7], new_db_nodpos[7]['nodpos']['data'][0] )

        # Check sand flags
        sand_classes = np.unique(np.concatenate(self.db.classes_of_state_variable("sand")))
        for class_name in sand_classes:
            sand_flags = reader.combine(new_db.query("sand", class_name, states=[1]))
            np.testing.assert_equal( sand_flags['sand']['data'][0], 1.0)
        new_db.close()

    #==============================================================================
    def test_copy_append_state_limit_states(self):
        """Tests copying existing database and then appending a state with state limit."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, merge_results=False )
        new_db.append_state( 0.0, limit_states_per_file=1 )
        new_db.append_state( 0.2, limit_states_per_file=1 )

        # Check that state maps have correct file numbers
        state_maps = new_db.state_maps()
        self.assertEqual(len(state_maps), 8)
        for processor_smap in state_maps:
            self.assertEqual( len(processor_smap), 2 )
            self.assertEqual( processor_smap[0].file_number, 0 )
            self.assertEqual( processor_smap[1].file_number, 1 )

    #==============================================================================
    def test_copy_append_state_limit_bytes(self):
        """Tests copying existing database and then appending a state with byte limit."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, merge_results=False )
        new_db.append_state( 0.0, limit_bytes_per_file=100 )
        new_db.append_state( 0.2, limit_bytes_per_file=100 )

        # Check that state maps have correct file numbers
        state_maps = new_db.state_maps()
        self.assertEqual(len(state_maps), 8)
        for processor_smap in state_maps:
            self.assertEqual( len(processor_smap), 2 )
            self.assertEqual( processor_smap[0].file_number, 0 )
            self.assertEqual( processor_smap[1].file_number, 1 )