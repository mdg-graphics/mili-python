#!/usr/bin/env python3
"""
SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili import reader
from mili.reductions import combine
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestCombineFunction(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

    def __compare_parallel_and_combined_result( self, parallel_result, combined_result ) -> bool:
        """Helper function to validate combine function."""
        for processor_result in parallel_result:
            for svar in processor_result:
                # Check svar exists
                self.assertTrue(svar in combined_result)
                # Check that states are the same
                if processor_result[svar]['layout']['states'].size != 0:
                    np.testing.assert_equal(processor_result[svar]['layout']['states'], combined_result[svar]['layout']['states'])
                for elem_index, element in enumerate(processor_result[svar]['layout']['labels']):
                    # Check that element exists
                    self.assertTrue( element in combined_result[svar]['layout']['labels'] )
                    combined_index = np.where( element == combined_result[svar]['layout']['labels'] )[0][0]
                    # Check that element result is the same for all states
                    for state_index, state in enumerate(processor_result[svar]['layout']['states']):
                        np.testing.assert_equal( processor_result[svar]['data'][state_index,elem_index,:], combined_result[svar]['data'][state_index,combined_index,:] )

    #==============================================================================
    def test_combine_scalar( self ):
        """ Test combine function with scalar result"""
        res = self.mili.query("sx", "brick", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_compile_multiple_scalars( self ):
        """ Test combine function with multiple scalar results"""
        res = self.mili.query(["sx","sy"], "brick", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_combine_vector( self ):
        """ Test combine function with vector result"""
        res = self.mili.query("stress", "brick", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_combine_vector_array( self ):
        """ Test combine function with vector array result"""
        res = self.mili.query("stress", "shell", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_combine_nodal_scalar( self ):
        """ Test combine function with nodal scalar result"""
        res = self.mili.query("ux", "node", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_combine_nodal_vector( self ):
        """ Test combine function with nodal vector result"""
        res = self.mili.query("nodpos", "node", states=[44, 78])
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    #==============================================================================
    def test_combine_multiple_glob_results( self ):
        """Test combine function on multiple global results"""
        res = self.mili.query(["pe", "he"], "glob")
        combined = combine( res )
        self.__compare_parallel_and_combined_result(res, combined)


class TestMergeDataFrames(unittest.TestCase):
    "Tests for the function merge_dataframes."
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.db = reader.open_database(TestMergeDataFrames.file_name, suppress_parallel=False, merge_results=False)

    def tearDown(self):
        self.db.close()

    #==============================================================================
    def test_single_scalar(self):
        sx_dict = combine(self.db.query("sx", "brick", states=[50,51,52,53,4], labels=[10,20,30]))
        sx_df = combine(self.db.query("sx", "brick", states=[50,51,52,53,4], labels=[10,20,30], as_dataframe=True))

        np.testing.assert_equal( sorted(sx_dict['sx']['layout']['labels']), sorted(list(sx_df['sx'].columns)) )
        np.testing.assert_equal( sorted(sx_dict['sx']['layout']['states']), sorted(list(sx_df['sx'].index)) )
        for i,ii in enumerate(sx_dict['sx']['layout']['states']):
            for j,jj in enumerate(sx_dict['sx']['layout']['labels']):
                np.testing.assert_equal(sx_dict['sx']['data'][i,j,:], sx_df['sx'][jj][ii])

    #==============================================================================
    def test_multiple_scalars(self):
        stress_dict = combine(self.db.query(["sx","sy"], "brick", states=[40,44,45,49], labels=[10,20,30]))
        stress_df = combine(self.db.query(["sx","sy"], "brick", states=[40,44,45,49], labels=[10,20,30], as_dataframe=True))

        np.testing.assert_equal( sorted(stress_dict['sx']['layout']['labels']), sorted(list(stress_df['sy'].columns)) )
        np.testing.assert_equal( sorted(stress_dict['sx']['layout']['states']), sorted(list(stress_df['sy'].index)) )
        for i,ii in enumerate(stress_dict['sx']['layout']['states']):
            for j,jj in enumerate(stress_dict['sx']['layout']['labels']):
                np.testing.assert_equal(stress_dict['sx']['data'][i,j,:], stress_df['sx'][jj][ii])

        np.testing.assert_equal( sorted(stress_dict['sy']['layout']['labels']), sorted(list(stress_df['sy'].columns)) )
        np.testing.assert_equal( sorted(stress_dict['sy']['layout']['states']), sorted(list(stress_df['sy'].index)) )
        for i,ii in enumerate(stress_dict['sy']['layout']['states']):
            for j,jj in enumerate(stress_dict['sy']['layout']['labels']):
                np.testing.assert_equal(stress_dict['sy']['data'][i,j,:], stress_df['sy'][jj][ii])

    #==============================================================================
    def test_vector(self):
        stress_dict = combine(self.db.query("stress", "brick", states=[30,31,32,33,34], labels=[20,30,33]))
        stress_df = combine(self.db.query("stress", "brick", states=[30,31,32,33,34], labels=[20,30,33], as_dataframe=True))

        np.testing.assert_equal( sorted(stress_dict['stress']['layout']['labels']), sorted(list(stress_df['stress'].columns)) )
        np.testing.assert_equal( sorted(stress_dict['stress']['layout']['states']), sorted(list(stress_df['stress'].index)) )
        for i,ii in enumerate(stress_dict['stress']['layout']['states']):
            for j,jj in enumerate(stress_dict['stress']['layout']['labels']):
                np.testing.assert_equal(stress_dict['stress']['data'][i,j,:], stress_df['stress'][jj][ii])

    #==============================================================================
    def test_vector_array(self):
        stress_dict = combine(self.db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44]))
        stress_df = combine(self.db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44], as_dataframe=True))

        np.testing.assert_equal( stress_dict['stress']['layout']['labels'], list(stress_df['stress'].columns) )
        np.testing.assert_equal( stress_dict['stress']['layout']['states'], list(stress_df['stress'].index) )
        for i,ii in enumerate(stress_df['stress'].index):
            for j,jj in enumerate(stress_df['stress'].columns):
                np.testing.assert_equal(stress_dict['stress']['data'][i,j,:], stress_df['stress'][jj][ii])

    #==============================================================================
    def test_node_scalar(self):
        """Test Nodal result to ensure we correctly handle duplicate elements."""
        ux_dict = combine(self.db.query("ux", "node", states=[10,20,30,40]))
        ux_df = combine(self.db.query("ux", "node", states=[10,20,30,40], as_dataframe=True))

        np.testing.assert_equal( ux_dict['ux']['layout']['labels'], sorted(list(ux_df['ux'].columns)) )
        np.testing.assert_equal( ux_dict['ux']['layout']['states'], sorted(list(ux_df['ux'].index)) )
        for i,ii in enumerate(ux_dict['ux']['layout']['states']):
            for j,jj in enumerate(ux_dict['ux']['layout']['labels']):
                np.testing.assert_equal(ux_dict['ux']['data'][i,j,:], ux_df['ux'][jj][ii])

    #==============================================================================
    def test_node_vector(self):
        """Test Nodal result to ensure we correctly handle duplicate elements."""
        nodpos_dict = combine(self.db.query("nodpos", "node", states=[10,20,30,40]))
        nodpos_df = combine(self.db.query("nodpos", "node", states=[10,20,30,40], as_dataframe=True))

        np.testing.assert_equal( nodpos_dict['nodpos']['layout']['labels'], sorted(list(nodpos_df['nodpos'].columns)) )
        np.testing.assert_equal( nodpos_dict['nodpos']['layout']['states'], sorted(list(nodpos_df['nodpos'].index)) )
        for i,ii in enumerate(nodpos_dict['nodpos']['layout']['states']):
            for j,jj in enumerate(nodpos_dict['nodpos']['layout']['labels']):
                np.testing.assert_equal(nodpos_dict['nodpos']['data'][i,j,:], nodpos_df['nodpos'][jj][ii])


class ReductionTests:
    class TestReductions(unittest.TestCase):

        #==============================================================================
        def test_append_state(self):
            new_db_name = f"{self._testMethodName}_d3samp6.plt"
            self.parallel.copy_non_state_data(new_db_name)

            new_db = reader.open_database( new_db_name,
                                        suppress_parallel=False, merge_results=True )

            parallel_result = new_db.append_state( 100.0 )
            self.assertEqual(parallel_result, 1)
            del new_db

            for proc in ["000", "001", "002", "003", "004", "005", "006", "007",""]:
                if os.path.exists(f"{self._testMethodName}_d3samp6.plt{proc}A"):
                    os.remove(f"{self._testMethodName}_d3samp6.plt{proc}A")
                if os.path.exists(f"{self._testMethodName}_d3samp6.plt{proc}00"):
                    os.remove(f"{self._testMethodName}_d3samp6.plt{proc}00")

        #==============================================================================
        def test_all_labels_of_material(self):
            for mat in [1,2,3]:
                serial_result = self.serial.all_labels_of_material(mat)
                parallel_result = self.parallel.all_labels_of_material(mat)

                # Check that same type is returned by both
                self.assertEqual( type(serial_result), type(parallel_result) )

                self.assertEqual(
                    sorted(list(serial_result.keys())),
                    sorted(list(parallel_result.keys()))
                )
                for class_name in serial_result:
                    sorted_serial = sorted(serial_result[class_name])
                    sorted_parallel = sorted(parallel_result[class_name])
                    np.testing.assert_equal(sorted_serial, sorted_parallel)

        #==============================================================================
        def test_containing_state_variables_of_class(self):
            serial_result = self.serial.containing_state_variables_of_class("sx", "brick")
            parallel_result = self.parallel.containing_state_variables_of_class("sx", "brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)

        #==============================================================================
        def test_copy_non_state_data(self):
            new_db_name = f"{self._testMethodName}_d3samp6.plt"
            parallel_result = self.parallel.copy_non_state_data(new_db_name)
            self.assertEqual(parallel_result, None)

            for proc in ["000", "001", "002", "003", "004", "005", "006", "007", ""]:
                if os.path.exists(f"{self._testMethodName}_d3samp6.plt{proc}A"):
                    os.remove(f"{self._testMethodName}_d3samp6.plt{proc}A")
                if os.path.exists(f"{self._testMethodName}_d3samp6.plt{proc}00"):
                    os.remove(f"{self._testMethodName}_d3samp6.plt{proc}00")

        #==============================================================================
        def test_class_labels_of_material(self):
            serial_result = self.serial.class_labels_of_material(2, "brick")
            parallel_result = self.parallel.class_labels_of_material(2, "brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_class_names(self):
            serial_result = self.serial.class_names()
            parallel_result = self.parallel.class_names()
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_classes_of_state_variable(self):
            serial_result = self.serial.classes_of_state_variable("sx")
            parallel_result = self.parallel.classes_of_state_variable("sx")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_state_variables_of_class(self):
            """
            NOTE: The state variables in the serial vs. parallel d3samp6 databases differ because
            the databases were generated with different version of the simulation codes. This is more
            of a regression tests that checks that this function runs
            """
            serial_result = self.serial.state_variables_of_class("glob")
            parallel_result = self.parallel.state_variables_of_class("glob")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )

            serial_result = self.serial.state_variables_of_class("mat")
            parallel_result = self.parallel.state_variables_of_class("mat")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )

            serial_result = self.serial.state_variables_of_class("brick")
            parallel_result = self.parallel.state_variables_of_class("brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )

        #==============================================================================
        def test_connectivity(self):
            for class_name in ["brick", "beam", "shell"]:
                serial_conns = self.serial.connectivity(class_name)
                parallel_conns = self.parallel.connectivity(class_name)
                # Check that same type is returned by both
                self.assertEqual( type(serial_conns), type(parallel_conns) )
                serial_labels = self.serial.labels(class_name)
                parallel_labels = self.parallel.labels(class_name)
                for idx, label in enumerate(serial_labels):
                    where = np.where(np.isin(parallel_labels, label))[0][0]
                    np.testing.assert_equal( serial_conns[idx], parallel_conns[where] )

            serial_conns = self.serial.connectivity()
            parallel_conns = self.parallel.connectivity()
            # Check that same type is returned by both
            self.assertEqual( type(serial_conns), type(parallel_conns) )
            np.testing.assert_equal( sorted(list(serial_conns.keys())), sorted(list(parallel_conns.keys())) )
            for class_name in serial_conns:
                serial_labels = self.serial.labels(class_name)
                parallel_labels = self.parallel.labels(class_name)
                for idx, label in enumerate(serial_labels):
                    where = np.where(np.isin(parallel_labels, label))[0][0]
                    np.testing.assert_equal( serial_conns[class_name][idx], parallel_conns[class_name][where] )

        #==============================================================================
        def test_components_of_vector_svar(self):
            serial_result = self.serial.components_of_vector_svar("stress")
            parallel_result = self.parallel.components_of_vector_svar("stress")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)

        #==============================================================================
        def test_classes_of_derived_variable(self):
            serial_result = self.serial.classes_of_derived_variable("eff_stress")
            parallel_result = self.parallel.classes_of_derived_variable("eff_stress")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_derived_variables_of_class(self):
            serial_result = self.serial.derived_variables_of_class("brick")
            parallel_result = self.parallel.derived_variables_of_class("brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_int_points_of_state_variable(self):
            serial_result = self.serial.int_points_of_state_variable("sx", "beam")
            parallel_result = self.parallel.int_points_of_state_variable("sx", "beam")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_integration_points(self):
            serial_result = self.serial.integration_points()
            parallel_result = self.parallel.integration_points()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(list(serial_result.keys())),
                sorted(list(parallel_result.keys()))
            )
            for mat in serial_result:
                np.testing.assert_equal(
                    sorted(serial_result[mat]),
                    sorted(parallel_result[mat])
                )

        #==============================================================================
        def test_labels(self):
            serial_labels = self.serial.labels()
            parallel_labels = self.parallel.labels()
            # Check that same type is returned by both
            self.assertEqual( type(serial_labels), type(parallel_labels) )
            np.testing.assert_equal(
                sorted(list(serial_labels.keys())),
                sorted(list(parallel_labels.keys()))
            )
            for class_name in serial_labels:
                np.testing.assert_equal(
                    sorted(serial_labels[class_name]),
                    sorted(parallel_labels[class_name])
                )

            serial_labels = self.serial.labels("node")
            parallel_labels = self.parallel.labels("node")
            # Check that same type is returned by both
            self.assertEqual( type(serial_labels), type(parallel_labels) )
            np.testing.assert_equal(
                sorted(serial_labels),
                sorted(parallel_labels)
            )

        #==============================================================================
        def test_material_classes(self):
            serial_result = self.serial.material_classes(2)
            parallel_result = self.parallel.material_classes(2)
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_material_numbers(self):
            serial_result = self.serial.material_numbers()
            parallel_result = self.parallel.material_numbers()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(
                sorted(serial_result),
                sorted(parallel_result)
            )

        #==============================================================================
        def test_materials(self):
            serial_materials = self.serial.materials()
            parallel_materials = self.parallel.materials()
            # Check that same type is returned by both
            self.assertEqual( type(serial_materials), type(parallel_materials) )
            np.testing.assert_equal(
                sorted(list(serial_materials.keys())),
                sorted(list(parallel_materials.keys()))
            )
            for mat_name in serial_materials:
                np.testing.assert_equal(
                    sorted(serial_materials[mat_name]),
                    sorted(parallel_materials[mat_name])
                )

        #==============================================================================
        def test_materials_of_class_name(self):
            serial_result = self.serial.materials_of_class_name("brick")
            parallel_result = self.parallel.materials_of_class_name("brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)

        #==============================================================================
        def test_mesh_dimensions(self):
            serial_result = self.serial.mesh_dimensions()
            parallel_result = self.parallel.mesh_dimensions()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            self.assertEqual(serial_result, parallel_result)

        #==============================================================================
        def test_nodes(self):
            serial_nodes = self.serial.nodes()
            parallel_nodes = self.parallel.nodes()
            self.assertEqual( type(serial_nodes), type(parallel_nodes) )
            self.assertEqual( len(serial_nodes), len(parallel_nodes) )
            serial_nlabels = self.serial.labels("node")
            parallel_nlabels = self.parallel.labels("node")
            np.testing.assert_equal( sorted(serial_nlabels), sorted(parallel_nlabels) )

            for idx, nlabel in enumerate(serial_nlabels):
                where = np.where(np.isin(parallel_nlabels, nlabel))[0][0]
                np.testing.assert_equal( serial_nodes[idx], parallel_nodes[where] )

        #==============================================================================
        def test_nodes_of_elems(self):
            serial_nodes, serial_labels = self.serial.nodes_of_elems("brick", [1,2,3,19,20,36])
            parallel_nodes, parallel_labels = self.parallel.nodes_of_elems("brick", [1,2,3,19,20,36])

            # Check that same type is returned by both
            self.assertEqual( type(serial_nodes), type(parallel_nodes) )
            self.assertEqual( type(serial_labels), type(parallel_labels) )

            np.testing.assert_equal( serial_labels, sorted(parallel_labels) )
            for idx, label in enumerate(serial_labels):
                where = np.where(np.isin(parallel_labels, label))[0]
                np.testing.assert_equal( parallel_nodes[where][0], serial_nodes[idx] )

        #==============================================================================
        def test_nodes_of_material(self):
            serial_result = self.serial.nodes_of_material(2)
            parallel_result = self.parallel.nodes_of_material(2)
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, sorted(parallel_result))

        #==============================================================================
        def test_parts_of_class_name(self):
            serial_result = self.serial.parts_of_class_name("brick")
            parallel_result = self.parallel.parts_of_class_name("brick")
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)

        #==============================================================================
        def test_queriable_svars(self):
            serial_result = self.serial.queriable_svars()
            parallel_result = self.parallel.queriable_svars()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            self.assertEqual( set(serial_result) - set(parallel_result), set() )

        #==============================================================================
        def test_state_variable_titles(self):
            serial_result = self.serial.state_variable_titles()
            parallel_result = self.parallel.state_variable_titles()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            for svar_name in serial_result:
                np.testing.assert_equal(
                    sorted(serial_result[svar_name]),
                    sorted(parallel_result[svar_name])
                )

        #==============================================================================
        def test_reload_state_maps(self):
            serial_result = self.serial.reload_state_maps()
            parallel_result = self.parallel.reload_state_maps()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            self.assertEqual(serial_result, parallel_result)

        #==============================================================================
        def test_srec_fmt_qty(self):
            serial_result = self.serial.srec_fmt_qty()
            parallel_result = self.parallel.srec_fmt_qty()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            self.assertEqual(serial_result, parallel_result)

        #==============================================================================
        def test_state_maps(self):
            serial_result = self.serial.state_maps()
            parallel_result = self.parallel.state_maps()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            # We can't compare the state maps exactly because the file offsets will be different
            # so we just check that they are the same length. test_times will check that the times
            # in the state maps match.
            self.assertEqual(len(serial_result), len(parallel_result))

        #==============================================================================
        def test_supported_derived_variables(self):
            serial_result = self.serial.supported_derived_variables()
            parallel_result = self.parallel.supported_derived_variables()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)

        #==============================================================================
        def test_times(self):
            serial_result = self.serial.times()
            parallel_result = self.parallel.times()
            # Check that same type is returned by both
            self.assertEqual( type(serial_result), type(parallel_result) )
            np.testing.assert_equal(serial_result, parallel_result)


class TestSerialReductions(ReductionTests.TestReductions):
    serial_file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.serial = reader.open_database( TestServerWrapperReductions.serial_file_name, suppress_parallel=True, merge_results=False)
        self.parallel = reader.open_database( TestServerWrapperReductions.serial_file_name, suppress_parallel=True, merge_results=True )

class TestServerWrapperReductions(ReductionTests.TestReductions):
    parallel_file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
    serial_file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.serial = reader.open_database( TestServerWrapperReductions.serial_file_name, suppress_parallel=True, merge_results=False)
        self.parallel = reader.open_database( TestServerWrapperReductions.parallel_file_name, suppress_parallel=False, merge_results=True )

    def tearDown(self):
        self.parallel.close()

class TestLoopWrapperReductions(ReductionTests.TestReductions):
    parallel_file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
    serial_file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.serial = reader.open_database( TestServerWrapperReductions.serial_file_name, suppress_parallel=True, merge_results=False)
        self.parallel = reader.open_database( TestServerWrapperReductions.parallel_file_name, suppress_parallel=True, merge_results=True )