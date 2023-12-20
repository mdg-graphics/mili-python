#!/usr/bin/env python3
"""
SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili import reader
from mili.utils import *
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestCombineFunction(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=True )

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

    def test_combine_scalar( self ):
        """ Test combine function with scalar result"""
        res = self.mili.query("sx", "brick", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    def test_compile_multiple_scalars( self ):
        """ Test combine function with multiple scalar results"""
        res = self.mili.query(["sx","sy"], "brick", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    def test_combine_vector( self ):
        """ Test combine function with vector result"""
        res = self.mili.query("stress", "brick", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    def test_combine_vector_array( self ):
        """ Test combine function with vector array result"""
        res = self.mili.query("stress", "shell", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    def test_combine_nodal_scalar( self ):
        """ Test combine function with nodal scalar result"""
        res = self.mili.query("ux", "node", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

    def test_combine_nodal_vector( self ):
        """ Test combine function with nodal vector result"""
        res = self.mili.query("nodpos", "node", states=[44, 78])
        combined = reader.combine( res )
        self.__compare_parallel_and_combined_result(res, combined)

class TestResultsByElementFunction(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=True )

    def __compare_parallel_and_reorganized_result( self, parallel_result, reorganized_result ) -> bool:
        """Helper function to validate results_by_element function."""
        for processor_result in parallel_result:
            for svar in processor_result:
                # Check svar exists
                self.assertTrue(svar in reorganized_result)

                for elem_index, element in enumerate(processor_result[svar]['layout']['labels']):
                    # Check that element exists
                    self.assertTrue( element in reorganized_result[svar] )
                    # Check that element result is the same for all states
                    for state_index, state in enumerate(processor_result[svar]['layout']['states']):
                        np.testing.assert_equal( processor_result[svar]['data'][state_index,elem_index,:], reorganized_result[svar][element][state_index] )

    def test_result_by_element_scalar( self ):
        """ Test result_by_element function with scalar result"""
        res = self.mili.query("sx", "brick", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

    def test_result_by_element_multiple_scalars( self ):
        """ Test result_by_element function with multiple scalar results"""
        res = self.mili.query(["sx","sy"], "brick", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

    def test_result_by_element_vector( self ):
        """ Test result_by_element function with vector result"""
        res = self.mili.query("stress", "brick", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

    def test_result_by_element_vector_array( self ):
        """ Test result_by_element function with vector array result"""
        res = self.mili.query("stress", "shell", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

    def test_result_by_element_nodal_scalar( self ):
        """ Test result_by_element function with nodal scalar result"""
        res = self.mili.query("ux", "node", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

    def test_result_by_element_nodal_vector( self ):
        """ Test result_by_element function with nodal vector result"""
        res = self.mili.query("nodpos", "node", states=[44, 78])
        reorganized = reader.results_by_element(res)
        self.__compare_parallel_and_reorganized_result(res, reorganized)

class TestQueryDataToDataFrame(unittest.TestCase):
    "Tests for the function ndarray_to_dataframe."
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def test_scalar(self):
        db = reader.open_database(TestQueryDataToDataFrame.file_name)
        result_dict = db.query("sx", "brick", states=[1,2,3,4,5], labels=[20,30,33])
        data = result_dict['sx']['data']
        states = result_dict['sx']['layout']['states']
        labels = result_dict['sx']['layout']['labels']
        df = query_data_to_dataframe(data, states, labels)

        np.testing.assert_equal(list(df.index), [1,2,3,4,5])
        np.testing.assert_equal(list(df.columns), [20,30,33])
        np.testing.assert_equal(data[0,:,0], df.iloc[0])
        np.testing.assert_equal(data[1,:,0], df.iloc[1])
        np.testing.assert_equal(data[2,:,0], df.iloc[2])
        np.testing.assert_equal(data[3,:,0], df.iloc[3])
        np.testing.assert_equal(data[4,:,0], df.iloc[4])

    def test_vector(self):
        db = reader.open_database(TestQueryDataToDataFrame.file_name)
        result_dict = db.query("stress", "brick", states=[1,2,3,4,5], labels=[20,30,33])
        data = result_dict['stress']['data']
        states = result_dict['stress']['layout']['states']
        labels = result_dict['stress']['layout']['labels']
        df = query_data_to_dataframe(data, states, labels)

        np.testing.assert_equal(list(df.index), [1,2,3,4,5])
        np.testing.assert_equal(list(df.columns), [20,30,33])
        for i in df.index:
            for j in df.columns:
                sidx = np.where(i == states)[0][0]
                nidx = np.where(j == labels)[0][0]
                np.testing.assert_equal(data[sidx,nidx,:], df.at[i,j])

class TestResultDictionaryToDataFrame(unittest.TestCase):
    "Tests for the function result_dictionary_to_dataframe."
    serial_db_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
    parallel_db_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def test_serial_single_scalar(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.serial_db_name)
        result_dict = db.query("sx", "brick", states=[1,2,3,4,5], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )

        df = new_res['sx']
        np.testing.assert_equal(list(df.index), [1,2,3,4,5])
        np.testing.assert_equal(list(df.columns), [20,30,33])
        np.testing.assert_equal(result_dict['sx']['data'][0,:,0], df.iloc[0])
        np.testing.assert_equal(result_dict['sx']['data'][1,:,0], df.iloc[1])
        np.testing.assert_equal(result_dict['sx']['data'][2,:,0], df.iloc[2])
        np.testing.assert_equal(result_dict['sx']['data'][3,:,0], df.iloc[3])
        np.testing.assert_equal(result_dict['sx']['data'][4,:,0], df.iloc[4])

    def test_serial_double_scalar(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.serial_db_name)
        result_dict = db.query(["sx","sy"], "brick", states=[1,3,5,9], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )

        sx_df = new_res['sx']
        np.testing.assert_equal(list(sx_df.index), [1,3,5,9])
        np.testing.assert_equal(list(sx_df.columns), [20,30,33])
        np.testing.assert_equal(result_dict['sx']['data'][0,:,0], sx_df.iloc[0])
        np.testing.assert_equal(result_dict['sx']['data'][1,:,0], sx_df.iloc[1])
        np.testing.assert_equal(result_dict['sx']['data'][2,:,0], sx_df.iloc[2])
        np.testing.assert_equal(result_dict['sx']['data'][3,:,0], sx_df.iloc[3])

        sy_df = new_res['sy']
        np.testing.assert_equal(list(sy_df.index), [1,3,5,9])
        np.testing.assert_equal(list(sy_df.columns), [20,30,33])
        np.testing.assert_equal(result_dict['sy']['data'][0,:,0], sy_df.iloc[0])
        np.testing.assert_equal(result_dict['sy']['data'][1,:,0], sy_df.iloc[1])
        np.testing.assert_equal(result_dict['sy']['data'][2,:,0], sy_df.iloc[2])
        np.testing.assert_equal(result_dict['sy']['data'][3,:,0], sy_df.iloc[3])

    def test_serial_vector(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.serial_db_name)
        result_dict = db.query("stress", "brick", states=[2,4,6,8], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )

        df = new_res['stress']
        np.testing.assert_equal(list(df.index), [2,4,6,8])
        np.testing.assert_equal(list(df.columns), [20,30,33])
        for i,ii in enumerate(df.index):
            for j,jj in enumerate(df.columns):
                np.testing.assert_equal(result_dict['stress']['data'][i,j,:], df.at[ii,jj])

    def test_serial_vector_array(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.serial_db_name)
        result_dict = db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44])
        new_res = result_dictionary_to_dataframe( result_dict )

        df = new_res['stress']
        np.testing.assert_equal(list(df.index), [10,20,30,40])
        np.testing.assert_equal(list(df.columns), [1,16,44])
        for i,ii in enumerate(df.index):
            for j,jj in enumerate(df.columns):
                np.testing.assert_equal(result_dict['stress']['data'][i,j,:], df.at[ii,jj])

    def test_parallel_single_scalar(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name, experimental=True)
        result_dict = db.query("sx", "brick", states=[1,2,3,4,5], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )
        combined_res = combine(result_dict)

        df = new_res['sx']
        np.testing.assert_equal(list(df.index), [1,2,3,4,5])
        np.testing.assert_equal(list(df.columns), [30,33,20])
        np.testing.assert_equal(combined_res['sx']['data'][0,:,0], df.iloc[0])
        np.testing.assert_equal(combined_res['sx']['data'][1,:,0], df.iloc[1])
        np.testing.assert_equal(combined_res['sx']['data'][2,:,0], df.iloc[2])
        np.testing.assert_equal(combined_res['sx']['data'][3,:,0], df.iloc[3])
        np.testing.assert_equal(combined_res['sx']['data'][4,:,0], df.iloc[4])
        db.close()

    def test_parallel_double_scalar(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name, experimental=True)
        result_dict = db.query(["sx","sy"], "brick", states=[1,3,5,9], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )
        combined_res = combine(result_dict)

        sx_df = new_res['sx']
        np.testing.assert_equal(list(sx_df.index), [1,3,5,9])
        np.testing.assert_equal(list(sx_df.columns), [30,33,20])
        np.testing.assert_equal(combined_res['sx']['data'][0,:,0], sx_df.iloc[0])
        np.testing.assert_equal(combined_res['sx']['data'][1,:,0], sx_df.iloc[1])
        np.testing.assert_equal(combined_res['sx']['data'][2,:,0], sx_df.iloc[2])
        np.testing.assert_equal(combined_res['sx']['data'][3,:,0], sx_df.iloc[3])

        sy_df = new_res['sy']
        np.testing.assert_equal(list(sy_df.index), [1,3,5,9])
        np.testing.assert_equal(list(sy_df.columns), [30,33,20])
        np.testing.assert_equal(combined_res['sy']['data'][0,:,0], sy_df.iloc[0])
        np.testing.assert_equal(combined_res['sy']['data'][1,:,0], sy_df.iloc[1])
        np.testing.assert_equal(combined_res['sy']['data'][2,:,0], sy_df.iloc[2])
        np.testing.assert_equal(combined_res['sy']['data'][3,:,0], sy_df.iloc[3])
        db.close()

    def test_parallel_vector(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name, experimental=True)
        result_dict = db.query("stress", "brick", states=[2,4,6,8], labels=[20,30,33])
        new_res = result_dictionary_to_dataframe( result_dict )
        combined_res = combine(result_dict)

        df = new_res['stress']
        np.testing.assert_equal(list(df.index), [2,4,6,8])
        np.testing.assert_equal(list(df.columns), [30,33,20])
        for i,ii in enumerate(df.index):
            for j,jj in enumerate(df.columns):
                np.testing.assert_equal(combined_res['stress']['data'][i,j,:], df.at[ii,jj])
        db.close()

    def test_parallel_vector_array(self):
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name, experimental=True)
        result_dict = db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44])
        new_res = result_dictionary_to_dataframe( result_dict )
        combined_res = combine(result_dict)

        df = new_res['stress']
        np.testing.assert_equal(list(df.index), [10,20,30,40])
        np.testing.assert_equal(list(df.columns), [1,16,44])
        for i,ii in enumerate(df.index):
            for j,jj in enumerate(df.columns):
                np.testing.assert_equal(combined_res['stress']['data'][i,j,:], df.at[ii,jj])
        db.close()

class TestMergeDataFrames(unittest.TestCase):
    "Tests for the function merge_dataframes."
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.db = reader.open_database(TestMergeDataFrames.file_name, experimental=True)

    def tearDown(self):
        self.db.close()

    def test_single_scalar(self):
        sx_dict = reader.combine(self.db.query("sx", "brick", states=[50,51,52,53,4], labels=[10,20,30]))
        sx_df = reader.combine(self.db.query("sx", "brick", states=[50,51,52,53,4], labels=[10,20,30], as_dataframe=True))

        np.testing.assert_equal( sorted(sx_dict['sx']['layout']['labels']), sorted(list(sx_df['sx'].columns)) )
        np.testing.assert_equal( sorted(sx_dict['sx']['layout']['states']), sorted(list(sx_df['sx'].index)) )
        for i,ii in enumerate(sx_dict['sx']['layout']['states']):
            for j,jj in enumerate(sx_dict['sx']['layout']['labels']):
                np.testing.assert_equal(sx_dict['sx']['data'][i,j,:], sx_df['sx'][jj][ii])

    def test_multiple_scalars(self):
        stress_dict = reader.combine(self.db.query(["sx","sy"], "brick", states=[40,44,45,49], labels=[10,20,30]))
        stress_df = reader.combine(self.db.query(["sx","sy"], "brick", states=[40,44,45,49], labels=[10,20,30], as_dataframe=True))

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

    def test_vector(self):
        stress_dict = reader.combine(self.db.query("stress", "brick", states=[30,31,32,33,34], labels=[20,30,33]))
        stress_df = reader.combine(self.db.query("stress", "brick", states=[30,31,32,33,34], labels=[20,30,33], as_dataframe=True))

        np.testing.assert_equal( sorted(stress_dict['stress']['layout']['labels']), sorted(list(stress_df['stress'].columns)) )
        np.testing.assert_equal( sorted(stress_dict['stress']['layout']['states']), sorted(list(stress_df['stress'].index)) )
        for i,ii in enumerate(stress_dict['stress']['layout']['states']):
            for j,jj in enumerate(stress_dict['stress']['layout']['labels']):
                np.testing.assert_equal(stress_dict['stress']['data'][i,j,:], stress_df['stress'][jj][ii])

    def test_vector_array(self):
        stress_dict = reader.combine(self.db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44]))
        stress_df = reader.combine(self.db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44], as_dataframe=True))

        np.testing.assert_equal( stress_dict['stress']['layout']['labels'], list(stress_df['stress'].columns) )
        np.testing.assert_equal( stress_dict['stress']['layout']['states'], list(stress_df['stress'].index) )
        for i,ii in enumerate(stress_df['stress'].index):
            for j,jj in enumerate(stress_df['stress'].columns):
                np.testing.assert_equal(stress_dict['stress']['data'][i,j,:], stress_df['stress'][jj][ii])

    def test_node_scalar(self):
        """Test Nodal result to ensure we correctly handle duplicate elements."""
        ux_dict = reader.combine(self.db.query("ux", "node", states=[10,20,30,40]))
        ux_df = reader.combine(self.db.query("ux", "node", states=[10,20,30,40], as_dataframe=True))

        np.testing.assert_equal( ux_dict['ux']['layout']['labels'], sorted(list(ux_df['ux'].columns)) )
        np.testing.assert_equal( ux_dict['ux']['layout']['states'], sorted(list(ux_df['ux'].index)) )
        for i,ii in enumerate(ux_dict['ux']['layout']['states']):
            for j,jj in enumerate(ux_dict['ux']['layout']['labels']):
                np.testing.assert_equal(ux_dict['ux']['data'][i,j,:], ux_df['ux'][jj][ii])

    def test_node_vector(self):
        """Test Nodal result to ensure we correctly handle duplicate elements."""
        nodpos_dict = reader.combine(self.db.query("nodpos", "node", states=[10,20,30,40]))
        nodpos_df = reader.combine(self.db.query("nodpos", "node", states=[10,20,30,40], as_dataframe=True))

        np.testing.assert_equal( nodpos_dict['nodpos']['layout']['labels'], sorted(list(nodpos_df['nodpos'].columns)) )
        np.testing.assert_equal( nodpos_dict['nodpos']['layout']['states'], sorted(list(nodpos_df['nodpos'].index)) )
        for i,ii in enumerate(nodpos_dict['nodpos']['layout']['states']):
            for j,jj in enumerate(nodpos_dict['nodpos']['layout']['labels']):
                np.testing.assert_equal(nodpos_dict['nodpos']['data'][i,j,:], nodpos_df['nodpos'][jj][ii])

class TestDataFrameToResultDictionary(unittest.TestCase):
    "Tests for the function result_dictionary_to_dataframe."
    db_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def test_single_scalar(self):
        db = reader.open_database(TestDataFrameToResultDictionary.db_name)
        result_dict = db.query("sx", "brick", states=[1,2,3,4,5], labels=[20,30,33], as_dataframe=False)
        result_df = db.query("sx", "brick", states=[1,2,3,4,5], labels=[20,30,33], as_dataframe=True)
        df_as_dict = dataframe_to_result_dictionary( result_df )

        np.testing.assert_equal( result_dict['sx']['layout']['states'], df_as_dict['sx']['layout']['states'])
        np.testing.assert_equal( result_dict['sx']['layout']['labels'], df_as_dict['sx']['layout']['labels'])
        np.testing.assert_equal( result_dict['sx']['data'], df_as_dict['sx']['data'])

    def test_double_scalar(self):
        db = reader.open_database(TestDataFrameToResultDictionary.db_name)
        result_dict = db.query(["sx","sy"], "brick", states=[1,3,5,9], labels=[20,30,33], as_dataframe=False)
        result_df = db.query(["sx","sy"], "brick", states=[1,3,5,9], labels=[20,30,33], as_dataframe=True)
        df_as_dict = dataframe_to_result_dictionary( result_df )

        np.testing.assert_equal( result_dict['sx']['layout']['states'], df_as_dict['sx']['layout']['states'])
        np.testing.assert_equal( result_dict['sx']['layout']['labels'], df_as_dict['sx']['layout']['labels'])
        np.testing.assert_equal( result_dict['sx']['data'], df_as_dict['sx']['data'])

        np.testing.assert_equal( result_dict['sy']['layout']['states'], df_as_dict['sy']['layout']['states'])
        np.testing.assert_equal( result_dict['sy']['layout']['labels'], df_as_dict['sy']['layout']['labels'])
        np.testing.assert_equal( result_dict['sy']['data'], df_as_dict['sy']['data'])

    def test_vector(self):
        db = reader.open_database(TestDataFrameToResultDictionary.db_name)
        result_dict = db.query("stress", "brick", states=[2,4,6,8], labels=[20,30,33], as_dataframe=False)
        result_df = db.query("stress", "brick", states=[2,4,6,8], labels=[20,30,33], as_dataframe=True)
        df_as_dict = dataframe_to_result_dictionary( result_df )

        np.testing.assert_equal( result_dict['stress']['layout']['states'], df_as_dict['stress']['layout']['states'])
        np.testing.assert_equal( result_dict['stress']['layout']['labels'], df_as_dict['stress']['layout']['labels'])
        np.testing.assert_equal( result_dict['stress']['data'], df_as_dict['stress']['data'])

    def test_vector_array(self):
        db = reader.open_database(TestDataFrameToResultDictionary.db_name)
        result_dict = db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44], as_dataframe=False)
        result_df = db.query("stress", "beam", states=[10,20,30,40], labels=[1,16,44], as_dataframe=True)
        df_as_dict = dataframe_to_result_dictionary( result_df )

        np.testing.assert_equal( result_dict['stress']['layout']['states'], df_as_dict['stress']['layout']['states'])
        np.testing.assert_equal( result_dict['stress']['layout']['labels'], df_as_dict['stress']['layout']['labels'])
        np.testing.assert_equal( result_dict['stress']['data'], df_as_dict['stress']['data'])