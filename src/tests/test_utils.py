#!/usr/bin/env python3
"""
SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili import reader
from mili.utils import result_dictionary_to_dataframe, query_data_to_dataframe, dataframe_to_result_dictionary
from mili.reductions import combine
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestResultsByElementFunction(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestResultsByElementFunction.file_name, suppress_parallel=True, merge_results=False )

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
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name)
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
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name)
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
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name)
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
        db = reader.open_database(TestResultDictionaryToDataFrame.parallel_db_name)
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