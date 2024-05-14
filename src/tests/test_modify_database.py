#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import os
import copy
import unittest
from mili import reader
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestModifySerialSingleState(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestModifySerialSingleState.file_name, suppress_parallel = True, merge_results=False )

    #==============================================================================
    def test_modify_state_variable(self):
        v1 = { 'matcgx' :
                { 'layout' :
                    {
                      'labels' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3,4], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 5.5 ] ], [ [ 4.5 ] ] ], dtype = np.float32)
                }
             }
        v2 = { 'matcgx' :
                { 'layout' :
                    {
                      'labels' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3,4], dtype = np.int32 )
                    },
                  'data' :  np.array([ [ [ 0.6021666526794434 ] ], [ [ 0.6021666526794434 ] ] ], dtype = np.float32)
                }
             }

        # Original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = [3,4] )
        self.assertEqual( answer['matcgx']['data'][0], 0.6021666526794434 )
        self.assertEqual( answer['matcgx']['data'][1], 0.6021666526794434 )

        # Modified
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = [3,4], write_data = v1 )
        self.assertEqual( answer['matcgx']['data'][0], 5.5 )
        self.assertEqual( answer['matcgx']['data'][1], 4.5 )

        # Back to original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = [3,4], write_data = v2 )
        self.assertEqual( answer['matcgx']['data'][0], 0.6021666526794434 )
        self.assertEqual( answer['matcgx']['data'][1], 0.6021666526794434 )

    #==============================================================================
    def test_modify_vector(self):
        v1 = { 'nodpos' :
                { 'layout' :
                    {
                        'labels' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [ 5.0, 6.0, 9.0 ], [ 5.1, 6.1, 9.1 ] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'nodpos' :
                { 'layout' :
                    {
                        'labels' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [0.4330127537250519, 0.2500000596046448, 2.436666965484619], [0.4330127239227295, 0.2499999850988388, 2.7033333778381348] ] ], dtype = np.float32 )
                }
             }

        # Before change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4 )
        np.testing.assert_equal( answer['nodpos']['data'][0,0,:], v2['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data'][0,1,:], v2['nodpos']['data'][0,1,:] )

        # After change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4, write_data = v1 )
        np.testing.assert_equal( answer['nodpos']['data'][0,0,:], v1['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data'][0,1,:], v1['nodpos']['data'][0,1,:] )

        # Back to original
        answer = self.mili.query('nodpos', 'node', labels = [70, 71],  states = 4, write_data = v2 )
        np.testing.assert_equal( answer['nodpos']['data'][0,0,:], v2['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data'][0,1,:], v2['nodpos']['data'][0,1,:] )

    #==============================================================================
    def test_modify_vector_component(self):
        v1 = { 'nodpos[uz]' :
                { 'layout' :
                    {
                        'labels' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [9.0], [9.0] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'nodpos[uz]' :
                { 'layout' :
                    {
                        'labels' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [2.436666965484619], [2.7033333778381348] ] ], dtype = np.float32 )
                }
             }

        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4 )
        self.assertEqual(answer['nodpos[uz]']['data'][0,0,0], 2.436666965484619)
        self.assertEqual(answer['nodpos[uz]']['data'][0,1,0], 2.7033333778381348)

        # After change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v1 )
        self.assertEqual(answer['nodpos[uz]']['data'][0,0,0], 9.0)
        self.assertEqual(answer['nodpos[uz]']['data'][0,1,0], 9.0)

        # Back to original
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v2 )
        self.assertEqual(answer['nodpos[uz]']['data'][0,0,0], 2.436666965484619)
        self.assertEqual(answer['nodpos[uz]']['data'][0,1,0], 2.7033333778381348)

    #==============================================================================
    def test_modify_vector_array(self):
        '''
        Test modifying a vector array
        '''
        v1 = { 'stress' :
                { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ -5547.1025390625, -5545.70751953125, -3.736035978363361e-07, 5546.4052734375, 0.4126972556114197, -0.412697434425354 ] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'stress' :
                { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ] ], dtype = np.float32 )
                }
             }
        # Before change
        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2 )
        np.testing.assert_equal( answer['stress']['data'], v1['stress']['data'] )

        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress']['data'], v2['stress']['data'] )

        # Back to original
        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress']['data'], v1['stress']['data'] )

    #==============================================================================
    def test_modify_vector_array_component(self):
        v1 = { 'stress[sy]' :
                 { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                   'data' : np.array( [ [ [ -5545.70751953125 ] ] ], dtype = np.float32 )
                 }
             }
        v2 = { 'stress[sy]' :
                 { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                   'data' : np.array( [ [ [ 12.0 ] ] ], dtype = np.float32 )
                 }
             }

        # Before change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        np.testing.assert_equal( answer['stress[sy]']['data'], v1['stress[sy]']['data'] )

        # After change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress[sy]']['data'], v2['stress[sy]']['data'] )

        # Back to original
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress[sy]']['data'], v1['stress[sy]']['data'] )


class TestModifySerialMultiState(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','mstate','d3samp6.plt_c')

    def setUp(self):
        self.mili = reader.open_database( TestModifySerialMultiState.file_name, suppress_parallel = True, merge_results=False )

    #==============================================================================
    def test_modify_vector_array(self):
        '''
        Test modifying a vector array
        '''
        v2 = { 'stress' :
                { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ] ], dtype = np.float32 )
                }
             }
        # Before change
        v1 = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2 )

        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress']['data'], v2['stress']['data'] )

        # Back to original
        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress']['data'], v1['stress']['data'] )

    #==============================================================================
    def test_modify_vector_array_component(self):
        '''
        Test modifying a vector array component
        '''
        v2 = { 'stress[sy]' :
                 { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                   'data' : np.array( [ [ [ 12.0 ] ] ], dtype = np.float32 )
                 }
             }

        # Before change
        v1 = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )

        # After change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress[sy]']['data'], v2['stress[sy]']['data'] )

        # Back to original
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress[sy]']['data'], v1['stress[sy]']['data'] )


class TestModifyParallelSingleState(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestModifyUncombinedDatabase.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

    #==============================================================================
    def test_modify_state_variable(self):
        v1 = { 'matcgx' :
                { 'layout' :
                    {
                      'labels' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 5.5 ] ] ], dtype = np.float32)
                }
             }
        v2 = { 'matcgx' :
                { 'layout' :
                    {
                      'labels' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 0.6021666526794434 ] ] ], dtype = np.float32)
                }
             }

        # Original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3 )
        self.assertEqual( answer[0]['matcgx']['data'][0], 0.6021666526794434 )

        # Modified
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v1 )
        self.assertEqual( answer[0]['matcgx']['data'][0], 5.5 )

        # Back to original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v2 )
        self.assertEqual( answer[0]['matcgx']['data'][0], 0.6021666526794434 )

    #==============================================================================
    def test_modify_vector(self):
        v1 = { 'nodpos' :
                { 'layout' :
                    {
                        'labels' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [ 5.0, 6.0, 9.0 ], [ 5.1, 6.1, 9.1 ] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'nodpos' :
                { 'layout' :
                    {
                        'labels' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [0.4330127537250519, 0.2500000596046448, 2.436666965484619], [0.4330127239227295, 0.2499999850988388, 2.7033333778381348] ] ], dtype = np.float32 )
                }
             }

        # Before change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4 )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,0,:], v2['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,1,:], v2['nodpos']['data'][0,1,:] )

        # After change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4, write_data = v1 )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,0,:], v1['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,1,:], v1['nodpos']['data'][0,1,:] )

        # Back to original
        answer = self.mili.query('nodpos', 'node', labels = [70, 71],  states = 4, write_data = v2 )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,0,:], v2['nodpos']['data'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data'][0,1,:], v2['nodpos']['data'][0,1,:] )

    #==============================================================================
    def test_modify_vector_component(self):
        v1 = { 'nodpos[uz]' :
                { 'layout' :
                    {
                        'labels' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [9.0], [9.0] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'nodpos[uz]' :
                { 'layout' :
                    {
                        'labels' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : np.array( [ [ [2.436666965484619], [2.7033333778381348] ] ], dtype = np.float32 )
                }
             }

        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4 )
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,0,0], 2.436666965484619)
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,1,0], 2.7033333778381348)

        # After change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v1 )
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,0,0], 9.0)
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,1,0], 9.0)

        # Back to original
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v2 )
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,0,0], 2.436666965484619)
        self.assertEqual(answer[0]['nodpos[uz]']['data'][0,1,0], 2.7033333778381348)

    #==============================================================================
    def test_modify_vector_array(self):
        v1 = { 'stress' :
                { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ -5377.6376953125, -5373.53173828125, -3.930831553589087e-07, 5375.58447265625, 0.6931889057159424, -0.693189263343811 ] ] ], dtype = np.float32 )
                }
             }
        v2 = { 'stress' :
                { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                  'data' : np.array([ [ [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] ] ], dtype = np.float32 )
                }
             }
        # Before change
        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2 )
        np.testing.assert_equal( answer[2]['stress']['data'], v1['stress']['data'] )

        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2 , write_data = v2 )
        np.testing.assert_equal( answer[2]['stress']['data'], v2['stress']['data'] )

        # Back to original
        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2 , write_data = v1 )
        np.testing.assert_equal( answer[2]['stress']['data'], v1['stress']['data'] )

    #==============================================================================
    def test_modify_vector_array_component(self):
        v1 = { 'stress[sy]' :
                 { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                   'data' : np.array( [ [ [ -5373.53173828125 ] ] ], dtype = np.float32 )
                 }
             }
        v2 = { 'stress[sy]' :
                 { 'layout' :
                    {
                        'labels' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                   'data' : np.array( [ [ [ 1.5 ] ] ], dtype = np.float32 )
                 }
             }

        # Before change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data'], v1['stress[sy]']['data'] )

        # After change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data'], v2['stress[sy]']['data'] )

        # Back to original
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data'], v1['stress[sy]']['data'] )


# TODO: Test with merge_results = True
class TestModifyUncombinedDatabase(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestModifyUncombinedDatabase.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

    #==============================================================================
    def test_modify_scalar(self):
        res = self.mili.query("sx", "brick", states=[35])
        original = reader.combine(res)

        # Modify database
        modified = copy.deepcopy(original)
        modified['sx']['data'][0,:] = 10.10
        res = self.mili.query("sx", "brick", states=[35], write_data=modified)
        np.testing.assert_equal( modified['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "brick", states=[35], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector(self):
        res = self.mili.query("stress", "brick", states=[35])
        original = reader.combine(res)

        # Modify database
        modified = copy.deepcopy(original)
        modified['stress']['data'][0,:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        res = self.mili.query("stress", "brick", states=[35], write_data=modified)
        np.testing.assert_equal( modified['stress']['data'], reader.combine(res)['stress']['data'] )

        # Back to original
        res = self.mili.query("stress", "brick", states=[35], write_data=original)
        np.testing.assert_equal( original['stress']['data'], reader.combine(res)['stress']['data'] )

    #==============================================================================
    def test_modify_vector_array_single_ipt(self):
        res = self.mili.query("sx", "beam", states=[35], ips=[1])
        original = reader.combine(res)

        # Modify database
        modified = copy.deepcopy(original)
        modified['sx']['data'][0,:] = 10.10
        res = self.mili.query("sx", "beam", states=[35], ips=[1], write_data=modified)
        np.testing.assert_equal( modified['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[35], ips=[1], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector_array_multiple_ipt(self):
        res = self.mili.query("sx", "beam", states=[35], ips=[1,3])
        original = reader.combine(res)

        # Modify database
        modified = copy.deepcopy(original)
        modified['sx']['data'][0,:] = [10.20, 30.40]
        res = self.mili.query("sx", "beam", states=[35], ips=[1,3], write_data=modified)
        np.testing.assert_equal( modified['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[35], ips=[1,3], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector_array_multiple_ipt_all(self):
        res = self.mili.query("sx", "beam", states=[35])
        original = reader.combine(res)

        # Modify database
        modified = copy.deepcopy(original)
        modified['sx']['data'][0,:] = [10.20, 30.40, 50.60, 70.80]
        res = self.mili.query("sx", "beam", states=[35], write_data=modified)
        np.testing.assert_equal( modified['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[35], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )


class TestModifyUncombinedDatabaseResultsByElement(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestModifyUncombinedDatabaseResultsByElement.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

    #==============================================================================
    def test_modify_scalar(self):
        res = self.mili.query("sx", "brick", states=[44])
        original = reader.combine(res)
        res_by_element = reader.results_by_element(res)

        # Modify database
        for elem in res_by_element['sx']:
            res_by_element['sx'][elem][:,:] = 10.10
        writeable = reader.writeable_from_results_by_element(res, res_by_element)
        res = self.mili.query("sx", "brick", states=[44], write_data=writeable)
        np.testing.assert_equal( writeable['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "brick", states=[44], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector(self):
        res = self.mili.query("stress", "brick", states=[44])
        original = reader.combine(res)
        res_by_element = reader.results_by_element(res)

        # Modify database
        for elem in res_by_element['stress']:
            res_by_element['stress'][elem][:,:] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        writeable = reader.writeable_from_results_by_element(res, res_by_element)
        res = self.mili.query("stress", "brick", states=[44], write_data=writeable)
        np.testing.assert_equal( writeable['stress']['data'], reader.combine(res)['stress']['data'] )

        # Back to original
        res = self.mili.query("stress", "brick", states=[44], write_data=original)
        np.testing.assert_equal( original['stress']['data'], reader.combine(res)['stress']['data'] )

    #==============================================================================
    def test_modify_vector_array_single_ipt(self):
        res = self.mili.query("sx", "beam", states=[44], ips=[1])
        original = reader.combine(res)
        res_by_element = reader.results_by_element(res)

        # Modify database
        for elem in res_by_element['sx']:
            res_by_element['sx'][elem][:,:] = 10.10
        writeable = reader.writeable_from_results_by_element(res, res_by_element)
        res = self.mili.query("sx", "beam", states=[44], ips=[1], write_data=writeable)
        np.testing.assert_equal( writeable['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[44], ips=[1], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector_array_multiple_ipt(self):
        res = self.mili.query("sx", "beam", states=[44], ips=[1,3])
        original = reader.combine(res)
        res_by_element = reader.results_by_element(res)

        # Modify database
        for elem in res_by_element['sx']:
            res_by_element['sx'][elem][:,:] = [10.20, 30.40]
        writeable = reader.writeable_from_results_by_element(res, res_by_element)
        res = self.mili.query("sx", "beam", states=[44], ips=[1,3], write_data=writeable)
        np.testing.assert_equal( writeable['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[44], ips=[1,3], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

    #==============================================================================
    def test_modify_vector_array_multiple_ipt_all(self):
        res = self.mili.query("sx", "beam", states=[44])
        original = reader.combine(res)
        res_by_element = reader.results_by_element(res)

        # Modify database
        for elem in res_by_element['sx']:
            res_by_element['sx'][elem][:,:] = [10.20, 30.40, 50.60, 70.80]
        writeable = reader.writeable_from_results_by_element(res, res_by_element)
        res = self.mili.query("sx", "beam", states=[44], write_data=writeable)
        np.testing.assert_equal( writeable['sx']['data'], reader.combine(res)['sx']['data'] )

        # Back to original
        res = self.mili.query("sx", "beam", states=[44], write_data=original)
        np.testing.assert_equal( original['sx']['data'], reader.combine(res)['sx']['data'] )

if __name__ == "__main__":
    unittest.main()
