#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import re
import os
import unittest
from mili import reader
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class DirectoryVersionTwo(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(DirectoryVersionTwo, self).__init__(*args,**kwargs)
        self.file_name = os.path.join(dir_path,'data','serial','dir_version_2','dblplt2009')

    def test_open( self ):
        mili = reader.open_database( self.file_name, suppress_parallel = True )

class NonsequentialMOBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NonsequentialMOBlocks, self).__init__(*args,**kwargs)
        self.file_name = os.path.join(dir_path,'data','parallel','basic1','basic1.plt')

    def test_serial( self ):
        mili = reader.open_database( self.file_name, procs = [7], suppress_parallel = True )
        result = mili.query( 'sx', 'brick', labels = [228], states = [10])
        self.assertAlmostEqual(result['sx']['data'][0][0][0], 20.355846, delta = 1e-6)

    def test_parallel( self ):
        mili = reader.open_database( self.file_name, merge_results=False )
        result = mili.query( 'sx', 'brick', labels = [228], states = [10] )
        self.assertAlmostEqual(result[7]['sx']['data'][0][0][0], 20.355846, delta = 1e-6 )

class NonMonotonicallyIncreasingMOBlocks(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(NonMonotonicallyIncreasingMOBlocks, self).__init__(*args,**kwargs)
        self.file_name = os.path.join(dir_path,'data','serial','vrt_BS','vrt_BS.plt')

    def test_query( self ):
        mili = reader.open_database( self.file_name, suppress_parallel = True )
        result = mili.query( 'refrcx', 'node', labels = [67], states = [5] )
        self.assertAlmostEqual(result['refrcx']['data'][0][0][0], 749.95404, delta = 1e-6 )

class NonsequentialMOBlocksTwo(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','fdamp1','fdamp1.plt')

    def setUp( self ):
        self.mili = reader.open_database( NonsequentialMOBlocksTwo.file_name, procs = [7], suppress_parallel = True )

    def test_refrcx( self ):
        result = self.mili.query( 'refrcx', 'node', labels = [6], states = [1,2,3])
        self.assertEqual(result['refrcx']['data'][0][0][0],          0.000000 )
        self.assertAlmostEqual(result['refrcx']['data'][1][0][0], -195.680618, delta = 1e-6 )
        self.assertAlmostEqual(result['refrcx']['data'][2][0][0], -374.033813, delta = 1e-6 )

class DoublePrecisionNodpos(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','beam_udi','beam_udi.plt')

    def setUp( self ):
        self.mili = reader.open_database( DoublePrecisionNodpos.file_name, suppress_parallel = True )

    def test_nodpos( self ):
        result = self.mili.query( 'nodpos', 'node', labels = [6], states = [3])
        self.assertAlmostEqual(result['nodpos']['data'][0][0][0], 499.86799237808793, delta = 1e-6 )
        self.assertAlmostEqual(result['nodpos']['data'][0][0][1], 100.00000000000000, delta = 1e-6 )
        self.assertAlmostEqual(result['nodpos']['data'][0][0][2], 198.08431992103525, delta = 1e-6 )

class QueryResultFromMultipleSubrecords(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','d3samp4','d3samp4.plt')

    def setUp( self ):
        self.mili = reader.open_database( VectorsInVectorArrays.file_name, suppress_parallel = True )

    def query_across_multiple_subrecords(self):
        """
        Tests bug in v0.4.0 where data array was created with incorrect size due to results
        being split over multiple subrecords
        """
        result = self.mili.query("mxx", "shell", states=[1])
        data_len = len(result['mxx']['data'][0])
        label_len = len(result['mxx']['layout']['labels'])
        self.assertEqual(data_len, label_len)

        # Elements 949 and 1431 are material 1, elements 4507 and 4518 are material 2,
        # and the results for mxx for each are stored in separate subrecords
        result = self.mili.query("mxx", "shell", states=[4], labels=[949, 1431, 4507, 4518])
        EXPECTED = [7.20788921E-06, 4.15517337E-04, -2.95172485E-05, -2.25825679E-05]
        self.assertAlmostEqual(result['mxx']['data'][0][0][0], EXPECTED[0], delta=2.0e-15)
        self.assertAlmostEqual(result['mxx']['data'][0][1][0], EXPECTED[1], delta=4.0e-13)
        self.assertAlmostEqual(result['mxx']['data'][0][2][0], EXPECTED[2], delta=5.0e-14)
        self.assertAlmostEqual(result['mxx']['data'][0][3][0], EXPECTED[3], delta=9.0e-15)

class InconsistantIntPointsForElementClassResult(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','basic1','basic1.plt')

    def setUp( self ):
        self.mili = reader.open_database( InconsistantIntPointsForElementClassResult.file_name )

    def query_inconsistant_int_points(self):
        """
        Test query for a result that has a different number of integration points in different subrecords.
        Material 5 has 8 int points
        Material 7 has 9 int points
        """
        with self.assertRaises(ValueError):
            result = self.mili.query("sx", "brick", states=[101], labels=[144,212])

        result = self.mili.query("sx", "brick", states=[101], labels=[144,212], ips=4)
        EXPECTED = [3.36948112e-02, 3.36948000e-02]
        self.assertAlmostEqual(result['sx']['data'][0][0][0], EXPECTED[0], delta=4.0e-11)
        self.assertAlmostEqual(result['sx']['data'][0][1][0], EXPECTED[1], delta=2.0e-11)

class VectorsInVectorArrays(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','d3samp4','d3samp4.plt')

    def setUp( self ):
        self.mili = reader.open_database( VectorsInVectorArrays.file_name, suppress_parallel = True )

    def test_query_scalar_in_vec_array(self):
        """
        Tests when the vector array has form [stress(vector), eps(scalar)].
        Verifies that the scalar can be correctly queried
        """
        result = self.mili.query( 'eps', 'shell', labels = [1], states = [2], ips = [1])
        self.assertAlmostEqual(result['eps']['data'][0][0][0], 2.3293568e-02 )
        result = self.mili.query( 'eps', 'shell', labels = [1], states = [2], ips = [2])
        self.assertAlmostEqual(result['eps']['data'][0][0][0], 7.1215495e-03 )

    def test_query_vector_in_vec_array(self):
        """
        Tests that the scalar components of a vector can be queried inside a vector array.
        """
        result = self.mili.query( 'sy', 'shell', labels = [24], states = [10], ips = [1])
        self.assertAlmostEqual(result['sy']['data'][0][0][0], -2.20756815e-03 )

        result = self.mili.query( 'sy', 'shell', labels = [24], states = [10], ips = [2])
        self.assertAlmostEqual(result['sy']['data'][0][0][0], 1.47148373e-03 )

        result = self.mili.query( 'sy', 'shell', labels = [24], states = [10])
        self.assertAlmostEqual(result['sy']['data'][0][0][0], -2.20756815e-03 )
        self.assertAlmostEqual(result['sy']['data'][0][0][1], 1.47148373e-03 )

        result = self.mili.query( 'syz', 'shell', labels = [24], states = [10], ips = [1])
        self.assertAlmostEqual(result['syz']['data'][0][0][0], -4.79422946e-04 )

        result = self.mili.query( 'syz', 'shell', labels = [24], states = [10], ips = [2])
        self.assertAlmostEqual(result['syz']['data'][0][0][0], 2.56596337e-04 )

        result = self.mili.query( 'syz', 'shell', labels = [24], states = [10])
        self.assertAlmostEqual(result['syz']['data'][0][0][0], -4.79422946e-04 )
        self.assertAlmostEqual(result['syz']['data'][0][0][1], 2.56596337e-04 )

        result = self.mili.query('stress', 'shell', labels = [15], states = [10], ips = 1)
        STRESS = np.array([ [ -2.1037722472e-03,
                              -2.4459683336e-03,
                              1.1138570699e-05,
                              2.3842934752e-04,
                              -4.1395323933e-05,
                              2.2738018743e-05 ] ], dtype = np.float32 )
        np.testing.assert_equal( result['stress']['data'][0,:,:], STRESS )

        # Test to make sure these run without exceptions
        result = self.mili.query('stress', 'shell', labels = [15], states = [10], ips = 2)
        result = self.mili.query('stress', 'shell', labels = [15], states = [10])
        result = self.mili.query('stress[sx]', 'shell', labels = [15], states = [10])
        result = self.mili.query('stress[syz]', 'shell', labels = [15], states = [10])

class Bugfixes0_2_4(unittest.TestCase):
    '''
    Testing labeling bugfixes from 0.2.4
    '''
    file_name = os.path.join(dir_path,'data','serial','labeling','dblplt003')
    def setUp(self):
        '''
        Set up the mili file object
        '''
        self.mili = reader.open_database( Bugfixes0_2_4.file_name, suppress_parallel=True )

    def test_labels_of_material(self):
        desired = np.array([
            22033, 21889, 21745, 21601, 21457, 21313, 21169, 21025, 20881,
            20737, 20593, 20449, 20305, 20161, 22034, 21890, 21746, 21602,
            21458, 21314, 21170, 21026, 20882, 20738, 20594, 20450, 20306,
            20162, 22035, 21891, 21747, 21603, 21459, 21315, 21171, 21027,
            20883, 20739, 20595, 20451, 20307, 20163, 22036, 21892, 21748,
            21604, 21460, 21316, 21172, 21028, 20884, 20740, 20596, 20452,
            20308, 20164, 22037, 21893, 21749, 21605, 21461, 21317, 21173,
            21029, 20885, 20741, 20597, 20453, 20309, 20165, 20021, 22038,
            21894, 21750, 21606, 21462, 21318, 21174, 21030, 20886, 20742,
            20598, 20454, 20310, 20166, 20022, 22039, 21895, 21751, 21607,
            21463, 21319, 21175, 21031, 20887, 20743, 20599, 20455, 20311,
            20167, 20023, 19879, 21896, 21752, 21608, 20600, 20456, 20312,
            20168, 20024, 21897, 21753 ], dtype=np.int32 )
        answer = self.mili.class_labels_of_material(6,'quad')
        np.testing.assert_equal( answer, desired )

class Bugfixes0_2_5(unittest.TestCase):
    '''
    Testing bugfixes from 0.2.5
    '''
    file_name = os.path.join(dir_path,'data','serial','vecarray','shell_mat15','shell_mat15.plt')
    '''
    Set up the mili file object
    '''
    def setUp(self):
        self.mili = reader.open_database( Bugfixes0_2_5.file_name, suppress_parallel=True )

    def test_strain_query(self):
        ''' prior to the 0.2.5 bugfixes, this returned 0,0,1e10,0,1e10,0 due to an srec offset error '''
        desired = { 'strain' :
                    { 'data' : np.array([[[0., 0., 0., 0., 0., 0.]]], dtype=np.float32),
                      'layout': { 'states': np.array([1], dtype=np.int32), 'labels': np.array([1], dtype=np.int32) },
                      'source': 'primal'
                    }
                  }
        answer = self.mili.query('strain','shell',labels=1,states=1)
        np.testing.assert_equal( answer, desired )

    def test_srec_offsets(self):
        '''
        our vec-array svar size calculation was off resulting in invalid srec offsets, this checks against
        the offsets directly from mili
        '''
        oracle_re = re.compile(r"([\w ]+):\s+Offset:\s+(\d+)\s+[\w\s]+=\s(\w+)")
        with open( os.path.join(dir_path,'data','serial','vecarray','shell_mat15','srec-offsets.txt') ) as fin:
            oracle_data = fin.read()
        desired = {}
        for srec_name, offset, _ in oracle_re.findall(oracle_data):
            desired[ srec_name ] = int( offset )
        result = {}
        for srec in self.mili._mili.subrecords():
            result[ srec.name ] = srec.state_byte_offset

        self.assertEqual( result, desired )

