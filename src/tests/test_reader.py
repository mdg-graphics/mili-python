#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import re
import os
import shutil
import copy
import unittest
from mili import reader
from mili.datatypes import Superclass
from mili.parallel import ReturnCode
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
        mili = reader.open_database( self.file_name )
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

class TestOpenDatabase(unittest.TestCase):
    def test_open_d3samp6_serial( self ):
        d3samp6 = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
        reader.open_database( d3samp6 )

class SerialSingleStateFile(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( SerialSingleStateFile.file_name, suppress_parallel = True )

    def test_invalid_inputs(self):
        '''
        Testing invalid inputs
        '''
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(4, 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
        with self.assertRaises(TypeError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
        with self.assertRaises(TypeError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 3, ips = 'cat')

    def test_nodes_getter(self):
        """
        Testing the getNodes() method of the Mili class.
        """
        NUM_NODES = 144
        FIRST_NODE = np.array( (1.0, 0.0, 0.0), np.float32 )
        LAST_NODE = np.array( (-6.556708598282057e-08, 1.5, 3.0), np.float32 )
        NODE9 = np.array( (0.5049999356269836, 0.8746857047080994, 1.0), np.float32 )
        NODE20 = np.array( (0.8677574396133423, 0.5009999871253967, 0.20000000298023224), np.float32 )
        NODE32 = np.array( (1.0019999742507935, 0.0, 1.7999999523162842), np.float32 )
        NODE54 = np.array( (0.3749999701976776, 0.6495190858840942, 2.0), np.float32 )
        NODE63 = np.array( (-6.556708598282057e-08, 1.5, 2.0), np.float32 )
        NODE88 = np.array( (0.3749999701976776, 0.6495190858840942, 2.200000047683716), np.float32 )
        NODE111 = np.array( (-4.371138828673793e-08, 1.0, 3.0), np.float32 )
        NODE124 = np.array( (-5.463923713477925e-08, 1.25, 2.200000047683716), np.float32 )

        nodes = self.mili.nodes()
        num_nodes = len(nodes)

        self.assertEqual(num_nodes, NUM_NODES)
        np.testing.assert_equal(nodes[0], FIRST_NODE)
        np.testing.assert_equal(nodes[num_nodes-1], LAST_NODE)
        np.testing.assert_equal(nodes[9], NODE9)
        np.testing.assert_equal(nodes[20], NODE20)
        np.testing.assert_equal(nodes[32], NODE32)
        np.testing.assert_equal(nodes[54], NODE54)
        np.testing.assert_equal(nodes[63], NODE63)
        np.testing.assert_equal(nodes[88], NODE88)
        np.testing.assert_equal(nodes[111], NODE111)
        np.testing.assert_equal(nodes[124], NODE124)

    def test_statemaps_getter(self):
        """
        Testing the state_maps() method of the Mili class
        """
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        state_maps = self.mili.state_maps()
        self.assertEqual(STATE_COUNT, len(state_maps))
        self.assertEqual(FIRST_STATE, state_maps[0].time)
        self.assertEqual(LAST_STATE, state_maps[-1].time)

    def test_times(self):
        """
        Testing the times() method of the Mili class
        """
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        times = self.mili.times()
        self.assertEqual(STATE_COUNT, len(times))
        self.assertEqual(FIRST_STATE, times[0])
        self.assertEqual(LAST_STATE, times[-1])

        times = self.mili.times([0,100])
        self.assertEqual(FIRST_STATE, times[0])
        self.assertEqual(LAST_STATE, times[-1])

    def test_labels_getter(self):
        """
        Testing the labels() method of the Mili class
        """
        labels = self.mili.labels()

        NUM_NODES = 144
        NUM_BEAM = 46
        NUM_BRICK = 36
        NUM_SHELL = 12
        NUM_GLOB = 1
        NUM_MATS = 5

        self.assertEqual(len(labels['node']), NUM_NODES)
        self.assertEqual(len(labels['beam']), NUM_BEAM)
        self.assertEqual(len(labels['brick']), NUM_BRICK)
        self.assertEqual(len(labels['shell']), NUM_SHELL)
        self.assertEqual(len(labels['glob']), NUM_GLOB)
        self.assertEqual(len(labels['mat']), NUM_MATS)

        NODE_LBLS = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                     13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                     23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                     33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
                     43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                     53, 54, 55, 56, 57, 58, 59, 60, 61, 62,
                     63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
                     73, 74, 75, 76, 77, 78, 79, 80, 81, 82,
                     83, 84, 85, 86, 87, 88, 89, 90, 91, 92,
                     93, 94, 95, 96, 97, 98, 99, 100, 101,
                     102, 103, 104, 105, 106, 107, 108, 109,
                     110, 111, 112, 113, 114, 115, 116, 117,
                     118, 119, 120, 121, 122, 123, 124, 125,
                     126, 127, 128, 129, 130, 131, 132, 133,
                     134, 135, 136, 137, 138, 139, 140, 141,
                     142, 143, 144 ] )
        BEAM_LBLS = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                     13, 14, 15, 16, 17, 18, 19, 20, 21,
                     22, 23, 24, 25, 26, 27, 28, 29, 30,
                     31, 32, 33, 34, 35, 36, 37, 38, 39,
                     40, 41, 42, 43, 44, 45, 46 ] )
        BRICK_LBLS = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
                      13, 14, 15, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26, 27, 28, 29, 30,
                      31, 32, 33, 34, 35, 36 ] )
        SHELL_LBLS = np.array( [ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 ] )
        GLOB_LBLS = np.array( [ 1 ] )
        MATS_LBLS = np.array( [ 1, 2, 3, 4, 5 ] )

        np.testing.assert_equal(labels['node'], NODE_LBLS)
        np.testing.assert_equal(labels['beam'], BEAM_LBLS)
        np.testing.assert_equal(labels['brick'], BRICK_LBLS)
        np.testing.assert_equal(labels['shell'], SHELL_LBLS)
        np.testing.assert_equal(labels['glob'], GLOB_LBLS)
        np.testing.assert_equal(labels['mat'], MATS_LBLS)

    def test_connectivity_with_material_number(self):
        """Test the connectivity method of the Mili class."""
        all_conn = self.mili.connectivity()
        conn_classes = list(all_conn.keys())
        self.assertEqual(conn_classes, ["beam", "brick", "shell", "cseg"])
        self.assertEqual(all_conn['beam'].shape, (46,4))
        self.assertEqual(all_conn['brick'].shape, (36,9))
        self.assertEqual(all_conn['shell'].shape, (12,5))
        self.assertEqual(all_conn['cseg'].shape, (24,5))

        self.assertTrue( all( all_conn['beam'][:,-1] == 1) )  # All beams are material 1
        self.assertTrue( all( all_conn['brick'][:,-1] == 2) )  # All bricks are material 2
        self.assertTrue( all( all_conn['shell'][:,-1] == 3) )  # All shells are material 3
        self.assertTrue( all( all_conn['cseg'][1:12,-1] == 4) )  # Csegs 1-12 are material 4
        self.assertTrue( all( all_conn['cseg'][12:24,-1] == 5) )  # Csegs 12-24 are material 5

    def test_reload_state_maps(self):
        """
        Test the reload_state_maps method of the Mili Class.
        """
        self.mili.reload_state_maps()
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        state_maps = self.mili.state_maps()
        self.assertEqual(STATE_COUNT, len(state_maps))
        self.assertEqual(FIRST_STATE, state_maps[0].time)
        self.assertEqual(LAST_STATE, state_maps[-1].time)

    def test_material_numbers(self):
        """
        Test getter for list of material numbers.
        """
        mat_nums = self.mili.material_numbers()
        self.assertEqual(set(mat_nums), set([1,2,3,4,5]))

    def test_classes_of_state_variable(self):
        """
        Test the classes_of_state_variable method of Mili Class.
        """
        sx_classes = self.mili.classes_of_state_variable('sx')
        self.assertEqual(set(sx_classes), set(["beam", "shell", "brick"]))

        uy_classes = self.mili.classes_of_state_variable('uy')
        self.assertEqual(uy_classes, ["node"])

        axf_classes = self.mili.classes_of_state_variable('axf')
        self.assertEqual(axf_classes, ["beam"])

    def test_materials_of_class_name(self):
        """
        Test the materials_of_class_name method of Mili Class.
        """
        brick_mats = self.mili.materials_of_class_name("brick")
        beam_mats = self.mili.materials_of_class_name("beam")
        shell_mats = self.mili.materials_of_class_name("shell")
        cseg_mats = self.mili.materials_of_class_name("cseg")

        self.assertEqual( brick_mats.size, 36 )
        np.testing.assert_equal( np.unique(brick_mats), np.array([2]) )

        self.assertEqual( beam_mats.size, 46 )
        np.testing.assert_equal( np.unique(beam_mats), np.array([1]) )

        self.assertEqual( shell_mats.size, 12 )
        np.testing.assert_equal( np.unique(shell_mats), np.array([3]) )

        self.assertEqual( cseg_mats.size, 24 )
        np.testing.assert_equal( cseg_mats, np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]) )

    def test_parts_of_class_name(self):
        """
        Test the parts_of_class_name method of Mili Class.
        """
        brick_parts = self.mili.parts_of_class_name("brick")
        beam_parts = self.mili.parts_of_class_name("beam")
        shell_parts = self.mili.parts_of_class_name("shell")
        cseg_parts = self.mili.parts_of_class_name("cseg")

        self.assertEqual( brick_parts.size, 36 )
        np.testing.assert_equal( np.unique(brick_parts), np.array([2]) )

        self.assertEqual( beam_parts.size, 46 )
        np.testing.assert_equal( np.unique(beam_parts), np.array([1]) )

        self.assertEqual( shell_parts.size, 12 )
        np.testing.assert_equal( np.unique(shell_parts), np.array([3]) )

        self.assertEqual( cseg_parts.size, 24 )
        np.testing.assert_equal( np.unique(cseg_parts), np.array([1]) )

    def test_mesh_object_classes_getter(self):
        """
        Test the mesh_object_classes() method of Mili Class.
        """
        mo_classes = self.mili.mesh_object_classes()

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 144)
        self.assertEqual(node_class.idents_exist, True)

        # beam class
        beam_class = mo_classes["beam"]
        self.assertEqual(beam_class.mesh_id, 0)
        self.assertEqual(beam_class.short_name, "beam")
        self.assertEqual(beam_class.long_name, "Beams")
        self.assertEqual(beam_class.sclass, Superclass.M_BEAM)
        self.assertEqual(beam_class.elem_qty, 46)
        self.assertEqual(beam_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 36)
        self.assertEqual(brick_class.idents_exist, True)

        # shell class
        shell_class = mo_classes["shell"]
        self.assertEqual(shell_class.mesh_id, 0)
        self.assertEqual(shell_class.short_name, "shell")
        self.assertEqual(shell_class.long_name, "Shells")
        self.assertEqual(shell_class.sclass, Superclass.M_QUAD)
        self.assertEqual(shell_class.elem_qty, 12)
        self.assertEqual(shell_class.idents_exist, True)

        # cseg class
        cseg_class = mo_classes["cseg"]
        self.assertEqual(cseg_class.mesh_id, 0)
        self.assertEqual(cseg_class.short_name, "cseg")
        self.assertEqual(cseg_class.long_name, "Contact Segment")
        self.assertEqual(cseg_class.sclass, Superclass.M_QUAD)
        self.assertEqual(cseg_class.elem_qty, 24)

    def test_statevariables_getter(self):
        """
        Testing state_variables()
        """
        state_variable_names = set(self.mili.state_variables().keys())
        SVAR_NAMES = set(['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
                          'tcon_fric_eng', 'tcon_eng', 'ew', 'te', 'rbvx', 'rbvy', 'rbvz', 'rbax',
                          'rbay', 'rbaz', 'init', 'plot', 'hsp', 'other_i_o', 'brick', 'beam', 'shell',
                          'tshell', 'discrete', 'delam', 'cohesive', 'ml', 'ntet', 'sph', 'kin_contact',
                          'reglag_contact', 'lag_solver', 'coupling', 'solution', 'xfem', 'total',
                          'cpu_time', 'matpe', 'matke', 'mathe', 'matbve', 'matdre', 'matstde',
                          'matflde', 'matte', 'matmass', 'matcgx', 'matcgy', 'matcgz', 'matxv', 'matyv',
                          'matzv', 'matxa', 'matya', 'matza', 'con_forx', 'con_fory', 'con_forz',
                          'con_momx', 'con_momy', 'con_momz', 'failure_bs', 'total_bs', 'cycles_bs',
                          'con_damp_eng', 'con_fric_eng', 'con_eng', 'sn', 'shmag', 'sr', 'ss', 's1',
                          's2', 's3', 'cseg_var', 'ux', 'uy', 'uz', 'nodpos', 'vx', 'vy', 'vz', 'nodvel',
                          'ax', 'ay', 'az', 'nodacc', 'sx', 'sy', 'sz', 'sxy', 'syz', 'szx', 'stress',
                          'eps', 'es_1a', 'ex', 'ey', 'ez', 'exy', 'eyz', 'ezx', 'strain', 'edrate',
                          'es_3a', 'es_3c', 'axf', 'sfs', 'sft', 'ms', 'mt', 'tor', 'max_eps', 'svec_x',
                          'svec_y', 'svec_z', 'svec', 'efs1', 'efs2', 'eps1', 'eps2', 'stress_mid',
                          'eeff_mid', 'stress_in', 'eeff_in', 'stress_out', 'eeff_out', 'mxx', 'myy',
                          'mxy', 'bend', 'qxx', 'qyy', 'shear', 'nxx', 'nyy', 'nxy', 'normal', 'thick',
                          'edv1', 'edv2', 'inteng', 'mid', 'in', 'out', 'press_cut', 'd_1', 'd_2',
                          'dam', 'frac_strain', 'sand', 'cause'])
        self.assertEqual(state_variable_names, SVAR_NAMES)

    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.all_labels_of_material('es_13')
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

       answer = self.mili.all_labels_of_material('3')
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

       answer = self.mili.all_labels_of_material(3)
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1')
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material('1')
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material(1)
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )

    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        nodes, elem_labels = self.mili.nodes_of_elems('brick', 1)
        self.assertEqual( elem_labels[0], 1 )
        self.assertEqual(nodes.size, 8)
        np.testing.assert_equal(nodes, np.array( [[65, 81, 85, 69, 66, 82, 86, 70]], dtype = np.int32 ))

    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query('matcgx', 'mat', labels = [1,2], states = 3 )
        self.assertEqual(answer['matcgx']['layout']['states'][0], 3)
        self.assertEqual(list(answer.keys()), ['matcgx'] )
        np.testing.assert_equal( answer['matcgx']['layout']['labels'], np.array( [ 1, 2 ], dtype = np.int32) )
        np.testing.assert_equal( answer['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32) )

    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query('nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual(answer['nodpos[ux]']['layout']['labels'][0], 70)
        self.assertEqual(answer['nodpos[ux]']['data'][0], 0.4330127537250519 )

        answer = self.mili.query('ux', 'node', labels = 70, states = 3 )
        self.assertEqual(answer['ux']['layout']['labels'][0], 70)
        self.assertEqual(answer['ux']['data'][0], 0.4330127537250519)

    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal( answer['nodpos']['data'][0,:,:], np.array( [ [ 0.4330127537250519, 0.2500000596046448, 2.436666965484619 ] ], dtype = np.float32 ) )

    '''
    Testing the modification of a scalar state variable
    '''
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

    '''
    Testing the modification of a vector state variable
    '''
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

    '''
    Testing the modification of a vector component
    '''
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

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307 ] ], dtype = np.float32 ) )

    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        self.assertEqual(answer['stress[sy]']['data'][0,0,0], -5545.70751953125)

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

    def test_modify_vector_array_component(self):
        '''
        Test modifying a vector array component
        '''
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

    def test_query_glob_results(self):
        """
        Test querying for results for M_MESH ("glob") element class.
        """
        answer = self.mili.query("he", "glob", states=[22])
        self.assertAlmostEqual( answer["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

        answer = self.mili.query("bve", "glob", states=[22])
        self.assertAlmostEqual( answer["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

        answer = self.mili.query("te", "glob", states=[22])
        self.assertAlmostEqual( answer["te"]["data"][0,0,0], 1629.718, delta=1e-4)

class SerialMutliStateFile(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','mstate','d3samp6.plt_c')

    def setUp( self ):
        self.mili = reader.open_database( SerialMutliStateFile.file_name )

    def test_statevariables_getter(self):
        """
        Testing state_variables()
        """
        state_variable_names = set(self.mili.state_variables().keys())
        SVAR_NAMES = set(['ax', 'axf', 'ay', 'az', 'beam', 'bend', 'brick', 'bve', 'cause', 'cohesive', 'con_damp_eng',
                          'con_eng', 'con_forx', 'con_fory', 'con_forz', 'con_fric_eng', 'con_momx', 'con_momy', 'con_momz',
                          'coupling', 'cpu_time', 'cseg_var', 'cw_cohesive', 'cw_coupling', 'cw_delam', 'cw_hsp', 'cw_main',
                          'cw_ml', 'cw_moment', 'cw_mpcheck', 'cw_other', 'cw_rigid_body', 'cw_slide', 'delam', 'discrete',
                          'dre', 'edrate', 'eps', 'es_1a', 'es_3a', 'es_3c', 'ew', 'ex', 'exy', 'ey', 'eyz', 'ez', 'ezx',
                          'flde', 'he', 'hsp', 'init', 'inteng', 'ke', 'ke_part', 'kin_contact', 'lag_solver', 'matbve',
                          'matcgx', 'matcgy', 'matcgz', 'matdre', 'matflde', 'mathe', 'matke', 'matmass', 'matpe', 'matstde',
                          'matte', 'matxa', 'matxv', 'matya', 'matyv', 'matza', 'matzv', 'ml', 'ms', 'mt', 'mxx', 'mxy', 'myy',
                          'nodacc', 'nodpos', 'nodvel', 'normal', 'ntet', 'nxx', 'nxy', 'nyy', 'other_i_o', 'pe', 'plot', 'qxx',
                          'qyy', 'rbax', 'rbay', 'rbaz', 'rbvx', 'rbvy', 'rbvz', 'reglag_contact', 's1', 's2', 's3', 'sand',
                          'sfs', 'sft', 'shear', 'shell', 'shmag', 'sn', 'solution', 'sph', 'sr', 'ss', 'stde', 'strain',
                          'stress', 'svec', 'svec_x', 'svec_y', 'svec_z', 'sx', 'sxy', 'sy', 'syz', 'sz', 'szx', 'tcon_damp_eng',
                          'tcon_eng', 'tcon_fric_eng', 'te', 'thick', 'tor', 'total', 'tshell', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz', 'xfem'])
        self.assertEqual(state_variable_names, SVAR_NAMES)

    def test_class_labels_of_mat(self):
        answer = self.mili.class_labels_of_material(5,'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

        answer = self.mili.class_labels_of_material("5",'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

        answer = self.mili.class_labels_of_material("slide1m",'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

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

    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        # almost equal, just down to float vs decimal repr
        self.assertAlmostEqual(answer['stress[sy]']['data'][0,0,0], -5373.5317, delta = 1e-4)

'''
Testing the parallel Mili file version
'''
class ParallelSingleStateFile(unittest.TestCase):
    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
    '''
    Set up the mili file object
    '''
    def setUp(self):
        self.mili = reader.open_database( ParallelSingleStateFile.file_name )

    def test_parallel_read(self):
        reader.open_database( ParallelSingleStateFile.file_name )

    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(4, 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
        with self.assertRaises(ValueError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
        with self.assertRaises(TypeError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
        with self.assertRaises(TypeError):
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 3, ips = 'cat')


    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
        nodes = self.mili.nodes()

        PROC_0_NODE_CNT = 35
        PROC_1_NODE_CNT = 40
        PROC_2_NODE_CNT = 26
        PROC_3_NODE_CNT = 36
        PROC_4_NODE_CNT = 20
        PROC_5_NODE_CNT = 27
        PROC_6_NODE_CNT = 18
        PROC_7_NODE_CNT = 18

        self.assertEqual(len(nodes[0]), PROC_0_NODE_CNT)
        self.assertEqual(len(nodes[1]), PROC_1_NODE_CNT)
        self.assertEqual(len(nodes[2]), PROC_2_NODE_CNT)
        self.assertEqual(len(nodes[3]), PROC_3_NODE_CNT)
        self.assertEqual(len(nodes[4]), PROC_4_NODE_CNT)
        self.assertEqual(len(nodes[5]), PROC_5_NODE_CNT)
        self.assertEqual(len(nodes[6]), PROC_6_NODE_CNT)
        self.assertEqual(len(nodes[7]), PROC_7_NODE_CNT)

    def test_statemaps_getter(self):
        """
        Testing the getStateMaps() method of the Mili class
        """
        state_maps = self.mili.state_maps()
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
        for state_map in state_maps:
            self.assertEqual(len(state_map), STATE_COUNT)
            self.assertEqual(state_map[0].time, FIRST_STATE)
            self.assertEqual(state_map[-1].time, LAST_STATE)

    def test_times(self):
        """
        Testing the times() method of the Mili class
        """
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        ptimes = self.mili.times()
        self.assertEqual(len(ptimes), PROCS)  # One entry in list for each processor
        for times in ptimes:
            self.assertEqual(STATE_COUNT, len(times))
            self.assertEqual(FIRST_STATE, times[0])
            self.assertEqual(LAST_STATE, times[-1])


    """
    Testing the getLabels() method of the Mili class
    """
    def test_labels_getter(self):
        result = self.mili.labels()
        goal = [
             {
              'brick': np.array([ 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 66,  67,  68,  70,  71,  72,  74,  75,  76,  78,  79,  80,  82, 83,  84,  86,  87,  88,  90,  91,  92,  94,  95,  96,  98,  99, 100, 102, 103, 104, 106, 107, 108, 111, 112], dtype=np.int32)
             },
             {
              'brick': np.array([ 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 90,  91,  94,  95,  98,  99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112, 114, 115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 128, 130, 131, 132, 134, 135, 136, 138, 139, 140, 142, 143, 144], dtype=np.int32)
             },
             {
              'beam': np.array([ 5, 6 ], dtype=np.int32),
              'brick': np.array([ 21, 23, 24 ], dtype=np.int32),
              'cseg': np.array([ 8, 9, 12, 20, 21, 24 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([  4,   6,  13,  14,  15,  58,  59,  60,  63,  64, 101, 102, 105, 106, 109, 110, 117, 118, 121, 122, 125, 126, 137, 138, 141, 142], dtype=np.int32),
              'shell': np.array([ 9, 11, 12 ], dtype=np.int32)
             },
             {
              'brick': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
              'cseg': np.array([ 1,  2,  3,  4,  5,  6, 13, 14, 15, 16, 17, 18 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 12,  13,  14,  15,  49,  50,  51,  52,  53,  54,  55,  56,  65, 66,  69,  70,  73,  74,  77,  78,  81,  82,  85,  86,  89,  90, 93,  94,  97,  98, 101, 102, 105, 106, 109, 110], dtype=np.int32),
              'shell': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
             },
             {
              'beam': np.array([ 2, 3, 12, 13, 17, 22, 23, 24, 25, 27, 28, 32, 33, 34 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 3,  4,  5,  6,  7, 12, 13, 16, 21, 22, 25, 29, 30, 31, 32, 33, 34, 37, 38, 39], dtype=np.int32)
             },
             {
              'beam': np.array([ 1, 4 ], dtype=np.int32),
              'brick': np.array([ 19, 20, 22 ], dtype=np.int32),
              'cseg': np.array([ 7, 10, 11, 19, 22, 23 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([  1,   2,   3,  12,  13,  57,  58,  59,  61,  62,  63,  97,  98, 101, 102, 113, 114, 117, 118, 121, 122, 129, 130, 133, 134, 137, 138], dtype=np.int32),
              'shell': np.array([ 7, 8, 10 ], dtype=np.int32)
             },
             {
              'beam': np.array([ 11, 18, 19, 20, 21, 29, 30, 31, 37, 38, 39, 40, 41, 42 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 8, 10, 14, 15, 16, 20, 25, 26, 27, 28, 34, 35, 36, 41, 42, 43, 44, 45], dtype=np.int32)
             },
             {
              'beam': np.array([ 7, 8, 9, 10, 14, 15, 16, 26, 35, 36, 43, 44, 45, 46 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 1,  9, 11, 16, 17, 18, 19, 20, 22, 23, 24, 32, 39, 40, 45, 46, 47, 48], dtype=np.int32)
             }
            ]

        for pr, gr in zip(result, goal):
            for sname, labels in pr.items():
                self.assertTrue( sname in gr.keys( ) )
                np.testing.assert_equal( labels, gr[sname] )

    # """
    # Testing the getMaterials() method of the Mili class.
    # """
    # def test_materials_getter(self):
    #     result = self.mili.labels_of_material(2)

    #     goal = [ {2: {'brick': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)}},
    #              {2: {'brick': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2], dtype=np.int32)}, 2: {'brick': np.array([1, 2, 3], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3], dtype=np.int32)}, 5: {'cseg': np.array([4, 5, 6], dtype=np.int32)}},
    #              {2: {'brick': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 5: {'cseg': np.array([7, 8, 9, 10, 11, 12], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2], dtype=np.int32)}, 2: {'brick': np.array([1, 2, 3], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3], dtype=np.int32)}, 5: {'cseg': np.array([4, 5, 6], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}} ]

    #     for pr, gr in zip(result,goal):
    #         for mat_num, mat in pr.items():
    #             self.assertTrue( mat_num in gr.keys() )
    #             for mat_name, labels, in mat.items():
    #                 self.assertTrue( mat_name in gr[mat_num].keys() )
    #                 np.testing.assert_equal( labels, gr[mat_num][mat_name])

    def test_reload_state_maps(self):
        """
        Test the reload_state_maps method of the Mili Class.
        """
        # Just make sure it doesn't cause Exception
        self.mili.reload_state_maps()
        state_maps = self.mili.state_maps()
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
        for state_map in state_maps:
            self.assertEqual(len(state_map), STATE_COUNT)
            self.assertEqual(state_map[0].time, FIRST_STATE)
            self.assertEqual(state_map[-1].time, LAST_STATE)

    def test_material_numbers(self):
        mat_nums = self.mili.material_numbers()
        self.assertEqual(mat_nums[0], [2])
        self.assertEqual(mat_nums[1], [2])
        self.assertEqual(mat_nums[2], [1,2,3,4,5])
        self.assertEqual(mat_nums[3], [2,3,4,5])
        self.assertEqual(mat_nums[4], [1])
        self.assertEqual(mat_nums[5], [1,2,3,4,5])
        self.assertEqual(mat_nums[6], [1])
        self.assertEqual(mat_nums[7], [1])

    def test_classes_of_state_variable(self):
        sx_classes = self.mili.classes_of_state_variable('sx')
        self.assertEqual(sx_classes[0], ["brick"])
        self.assertEqual(sx_classes[1], ["brick"])
        self.assertEqual(set(sx_classes[2]), set(["beam", "shell", "brick"]))
        self.assertEqual(set(sx_classes[3]), set(["shell", "brick"]))
        self.assertEqual(sx_classes[4], ["beam"])
        self.assertEqual(set(sx_classes[5]), set(["beam", "shell", "brick"]))
        self.assertEqual(sx_classes[6], ["beam"])
        self.assertEqual(sx_classes[7], ["beam"])

        uy_classes = self.mili.classes_of_state_variable('uy')
        self.assertEqual(uy_classes[0], ["node"])
        self.assertEqual(uy_classes[1], ["node"])
        self.assertEqual(uy_classes[2], ["node"])
        self.assertEqual(uy_classes[3], ["node"])
        self.assertEqual(uy_classes[4], ["node"])
        self.assertEqual(uy_classes[5], ["node"])
        self.assertEqual(uy_classes[6], ["node"])
        self.assertEqual(uy_classes[7], ["node"])

        axf_classes = self.mili.classes_of_state_variable('axf')
        self.assertEqual(axf_classes[0], [])
        self.assertEqual(axf_classes[1], [])
        self.assertEqual(axf_classes[2], ["beam"])
        self.assertEqual(axf_classes[3], [])
        self.assertEqual(axf_classes[4], ["beam"])
        self.assertEqual(axf_classes[5], ["beam"])
        self.assertEqual(axf_classes[6], ["beam"])
        self.assertEqual(axf_classes[7], ["beam"])

    def test_materials_of_class_name(self):
        """
        Test the materials_of_class_name method of Mili Class.
        """
        brick_mats = self.mili.materials_of_class_name("brick")
        beam_mats = self.mili.materials_of_class_name("beam")
        shell_mats = self.mili.materials_of_class_name("shell")
        cseg_mats = self.mili.materials_of_class_name("cseg")

        self.assertEqual( brick_mats[0].size, 11 )
        self.assertEqual( brick_mats[1].size, 13 )
        self.assertEqual( brick_mats[2].size, 3 )
        self.assertEqual( brick_mats[3].size, 6 )
        self.assertEqual( brick_mats[4].size, 0 )
        self.assertEqual( brick_mats[5].size, 3 )
        self.assertEqual( brick_mats[6].size, 0 )
        self.assertEqual( brick_mats[7].size, 0 )
        np.testing.assert_equal( np.unique(brick_mats[0]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[1]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[2]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[3]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[5]), np.array([2]) )

        self.assertEqual( beam_mats[0].size, 0 )
        self.assertEqual( beam_mats[1].size, 0 )
        self.assertEqual( beam_mats[2].size, 2 )
        self.assertEqual( beam_mats[3].size, 0 )
        self.assertEqual( beam_mats[4].size, 14 )
        self.assertEqual( beam_mats[5].size, 2 )
        self.assertEqual( beam_mats[6].size, 14 )
        self.assertEqual( beam_mats[7].size, 14 )
        np.testing.assert_equal( np.unique(beam_mats[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[4]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[5]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[6]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[7]), np.array([1]) )

        self.assertEqual( shell_mats[0].size, 0 )
        self.assertEqual( shell_mats[1].size, 0 )
        self.assertEqual( shell_mats[2].size, 3 )
        self.assertEqual( shell_mats[3].size, 6 )
        self.assertEqual( shell_mats[4].size, 0 )
        self.assertEqual( shell_mats[5].size, 3 )
        self.assertEqual( shell_mats[6].size, 0 )
        self.assertEqual( shell_mats[7].size, 0 )
        np.testing.assert_equal( np.unique(shell_mats[2]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_mats[3]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_mats[5]), np.array([3]) )

        self.assertEqual( cseg_mats[0].size, 0 )
        self.assertEqual( cseg_mats[1].size, 0 )
        self.assertEqual( cseg_mats[2].size, 6 )
        self.assertEqual( cseg_mats[3].size, 12 )
        self.assertEqual( cseg_mats[4].size, 0 )
        self.assertEqual( cseg_mats[5].size, 6 )
        self.assertEqual( cseg_mats[6].size, 0 )
        self.assertEqual( cseg_mats[7].size, 0 )
        np.testing.assert_equal( cseg_mats[2], np.array([4, 4, 4, 5, 5, 5]) )
        np.testing.assert_equal( cseg_mats[3], np.array([4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]) )
        np.testing.assert_equal( cseg_mats[5], np.array([4, 4, 4, 5, 5, 5]))


    def test_parts_of_class_name(self):
        """
        Test the parts_of_class_name method of Mili Class.
        """
        brick_parts = self.mili.parts_of_class_name("brick")
        beam_parts = self.mili.parts_of_class_name("beam")
        shell_parts = self.mili.parts_of_class_name("shell")
        cseg_parts = self.mili.parts_of_class_name("cseg")

        self.assertEqual( brick_parts[0].size, 11 )
        self.assertEqual( brick_parts[1].size, 13 )
        self.assertEqual( brick_parts[2].size, 3 )
        self.assertEqual( brick_parts[3].size, 6 )
        self.assertEqual( brick_parts[4].size, 0 )
        self.assertEqual( brick_parts[5].size, 3 )
        self.assertEqual( brick_parts[6].size, 0 )
        self.assertEqual( brick_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(brick_parts[0]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[1]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[2]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[3]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[5]), np.array([2]) )

        self.assertEqual( beam_parts[0].size, 0 )
        self.assertEqual( beam_parts[1].size, 0 )
        self.assertEqual( beam_parts[2].size, 2 )
        self.assertEqual( beam_parts[3].size, 0 )
        self.assertEqual( beam_parts[4].size, 14 )
        self.assertEqual( beam_parts[5].size, 2 )
        self.assertEqual( beam_parts[6].size, 14 )
        self.assertEqual( beam_parts[7].size, 14 )
        np.testing.assert_equal( np.unique(beam_parts[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[4]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[5]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[6]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[7]), np.array([1]) )

        self.assertEqual( shell_parts[0].size, 0 )
        self.assertEqual( shell_parts[1].size, 0 )
        self.assertEqual( shell_parts[2].size, 3 )
        self.assertEqual( shell_parts[3].size, 6 )
        self.assertEqual( shell_parts[4].size, 0 )
        self.assertEqual( shell_parts[5].size, 3 )
        self.assertEqual( shell_parts[6].size, 0 )
        self.assertEqual( shell_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(shell_parts[2]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_parts[3]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_parts[5]), np.array([3]) )

        self.assertEqual( cseg_parts[0].size, 0 )
        self.assertEqual( cseg_parts[1].size, 0 )
        self.assertEqual( cseg_parts[2].size, 6 )
        self.assertEqual( cseg_parts[3].size, 12 )
        self.assertEqual( cseg_parts[4].size, 0 )
        self.assertEqual( cseg_parts[5].size, 6 )
        self.assertEqual( cseg_parts[6].size, 0 )
        self.assertEqual( cseg_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(cseg_parts[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(cseg_parts[3]), np.array([1]) )
        np.testing.assert_equal( np.unique(cseg_parts[5]), np.array([1]) )


    def test_mesh_object_classes_getter(self):
        """
        Test the mesh_object_classes() method of Mili Class.
        """
        MO_classes = self.mili.mesh_object_classes()

        # Test 0th processor
        mo_classes = MO_classes[0]

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 35)
        self.assertEqual(node_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 11)
        self.assertEqual(brick_class.idents_exist, True)

        # Test processor 5
        mo_classes = MO_classes[5]

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 27)
        self.assertEqual(node_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 3)
        self.assertEqual(brick_class.idents_exist, True)

        # shell class
        shell_class = mo_classes["shell"]
        self.assertEqual(shell_class.mesh_id, 0)
        self.assertEqual(shell_class.short_name, "shell")
        self.assertEqual(shell_class.long_name, "Shells")
        self.assertEqual(shell_class.sclass, Superclass.M_QUAD)
        self.assertEqual(shell_class.elem_qty, 3)
        self.assertEqual(shell_class.idents_exist, True)

        # cseg class
        cseg_class = mo_classes["cseg"]
        self.assertEqual(cseg_class.mesh_id, 0)
        self.assertEqual(cseg_class.short_name, "cseg")
        self.assertEqual(cseg_class.long_name, "Contact Segment")
        self.assertEqual(cseg_class.sclass, Superclass.M_QUAD)
        self.assertEqual(cseg_class.elem_qty, 6)

    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        result = self.mili.state_variables()
        goal = [159,126,134,134,126,134,126,126]
        result = [len(rr) for rr in result]
        self.assertEqual( result, goal )

    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.all_labels_of_material('es_13')
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

       answer = self.mili.all_labels_of_material('3')
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

       answer = self.mili.all_labels_of_material(3)
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))


    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1')
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material('1')
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material(1)
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elems('brick', 1)
        np.testing.assert_equal( answer[3][0][0,:], np.array([65,81,85,69,66,82,86,70],dtype=np.int32) )


    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query( 'matcgx', 'mat', labels = [1,2], states = 3 )
        np.testing.assert_equal( answer[0]['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32 ) )

    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query( 'nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['nodpos[ux]']['data'][0,0,0], 0.4330127537250519)
        answer = self.mili.query( 'ux', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['ux']['data'][0,0,0], 0.4330127537250519)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal(answer[3]['nodpos']['data'][0,0,:], np.array( [0.4330127537250519, 0.2500000596046448, 2.436666965484619], dtype = np.float32 ) )

    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )

        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )


    '''
    Testing the modification of a scalar state variable
    '''
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

    '''
    Testing the modification of a vector state variable
    '''
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

    '''
    Testing the modification of a vector component
    '''
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

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer[2]['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307] ], dtype = np.float32 ) )

    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        self.assertEqual(answer[2]['stress[sy]']['data'][0,0,0], -5373.53173828125)

    '''
    Test modifying a vector array
    '''
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
    '''
    Test modifying a vector array component
    '''
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

    def test_query_glob_results(self):
        """
        Test querying for results for M_MESH ("glob") element class.
        """
        answer = self.mili.query("he", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

        answer = self.mili.query("bve", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

        answer = self.mili.query("te", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["te"]["data"][0,0,0], 1629.718, delta=1e-4)


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

    def test_nodpres_query(self):
        '''
        this was erroring on the provided database due to srec scalar svar "coord" lookup,
        so as long as it doesn't throw an exception we're good
        '''
        self.mili.query('nodpres','node')

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
        for srec in self.mili._MiliDatabase__srecs:
            result[ srec.name ] = srec.state_byte_offset

        self.assertEqual( result, desired )


'''
Testing the Experimental parallel Mili file version
'''
class ExperimentalParallelSingleStateFile(unittest.TestCase):
    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')
    '''
    Set up the mili file object
    '''
    def setUp(self):
        self.mili = reader.open_database( ParallelSingleStateFile.file_name, experimental=True )

    def tearDown(self):
        self.mili.close()

    def test_parallel_read(self):
        reader.open_database( ParallelSingleStateFile.file_name, experimental=True )

    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
        nodes = self.mili.nodes()

        PROC_0_NODE_CNT = 35
        PROC_1_NODE_CNT = 40
        PROC_2_NODE_CNT = 26
        PROC_3_NODE_CNT = 36
        PROC_4_NODE_CNT = 20
        PROC_5_NODE_CNT = 27
        PROC_6_NODE_CNT = 18
        PROC_7_NODE_CNT = 18

        self.assertEqual(len(nodes[0]), PROC_0_NODE_CNT)
        self.assertEqual(len(nodes[1]), PROC_1_NODE_CNT)
        self.assertEqual(len(nodes[2]), PROC_2_NODE_CNT)
        self.assertEqual(len(nodes[3]), PROC_3_NODE_CNT)
        self.assertEqual(len(nodes[4]), PROC_4_NODE_CNT)
        self.assertEqual(len(nodes[5]), PROC_5_NODE_CNT)
        self.assertEqual(len(nodes[6]), PROC_6_NODE_CNT)
        self.assertEqual(len(nodes[7]), PROC_7_NODE_CNT)

    def test_statemaps_getter(self):
        """
        Testing the getStateMaps() method of the Mili class
        """
        state_maps = self.mili.state_maps()
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
        for state_map in state_maps:
            self.assertEqual(len(state_map), STATE_COUNT)
            self.assertEqual(state_map[0].time, FIRST_STATE)
            self.assertEqual(state_map[-1].time, LAST_STATE)

    def test_times(self):
        """
        Testing the times() method of the Mili class
        """
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        ptimes = self.mili.times()
        self.assertEqual(len(ptimes), PROCS)  # One entry in list for each processor
        for times in ptimes:
            self.assertEqual(STATE_COUNT, len(times))
            self.assertEqual(FIRST_STATE, times[0])
            self.assertEqual(LAST_STATE, times[-1])


    """
    Testing the getLabels() method of the Mili class
    """
    def test_labels_getter(self):
        result = self.mili.labels()
        goal = [
             {
              'brick': np.array([ 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 66,  67,  68,  70,  71,  72,  74,  75,  76,  78,  79,  80,  82, 83,  84,  86,  87,  88,  90,  91,  92,  94,  95,  96,  98,  99, 100, 102, 103, 104, 106, 107, 108, 111, 112], dtype=np.int32)
             },
             {
              'brick': np.array([ 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 90,  91,  94,  95,  98,  99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112, 114, 115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 128, 130, 131, 132, 134, 135, 136, 138, 139, 140, 142, 143, 144], dtype=np.int32)
             },
             {
              'beam': np.array([ 5, 6 ], dtype=np.int32),
              'brick': np.array([ 21, 23, 24 ], dtype=np.int32),
              'cseg': np.array([ 8, 9, 12, 20, 21, 24 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([  4,   6,  13,  14,  15,  58,  59,  60,  63,  64, 101, 102, 105, 106, 109, 110, 117, 118, 121, 122, 125, 126, 137, 138, 141, 142], dtype=np.int32),
              'shell': np.array([ 9, 11, 12 ], dtype=np.int32)
             },
             {
              'brick': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
              'cseg': np.array([ 1,  2,  3,  4,  5,  6, 13, 14, 15, 16, 17, 18 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 12,  13,  14,  15,  49,  50,  51,  52,  53,  54,  55,  56,  65, 66,  69,  70,  73,  74,  77,  78,  81,  82,  85,  86,  89,  90, 93,  94,  97,  98, 101, 102, 105, 106, 109, 110], dtype=np.int32),
              'shell': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
             },
             {
              'beam': np.array([ 2, 3, 12, 13, 17, 22, 23, 24, 25, 27, 28, 32, 33, 34 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 3,  4,  5,  6,  7, 12, 13, 16, 21, 22, 25, 29, 30, 31, 32, 33, 34, 37, 38, 39], dtype=np.int32)
             },
             {
              'beam': np.array([ 1, 4 ], dtype=np.int32),
              'brick': np.array([ 19, 20, 22 ], dtype=np.int32),
              'cseg': np.array([ 7, 10, 11, 19, 22, 23 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([  1,   2,   3,  12,  13,  57,  58,  59,  61,  62,  63,  97,  98, 101, 102, 113, 114, 117, 118, 121, 122, 129, 130, 133, 134, 137, 138], dtype=np.int32),
              'shell': np.array([ 7, 8, 10 ], dtype=np.int32)
             },
             {
              'beam': np.array([ 11, 18, 19, 20, 21, 29, 30, 31, 37, 38, 39, 40, 41, 42 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 8, 10, 14, 15, 16, 20, 25, 26, 27, 28, 34, 35, 36, 41, 42, 43, 44, 45], dtype=np.int32)
             },
             {
              'beam': np.array([ 7, 8, 9, 10, 14, 15, 16, 26, 35, 36, 43, 44, 45, 46 ], dtype=np.int32),
              'glob': np.array([1], dtype=np.int32),
              'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
              'node': np.array([ 1,  9, 11, 16, 17, 18, 19, 20, 22, 23, 24, 32, 39, 40, 45, 46, 47, 48], dtype=np.int32)
             }
            ]

        for pr, gr in zip(result, goal):
            for sname, labels in pr.items():
                self.assertTrue( sname in gr.keys( ) )
                np.testing.assert_equal( labels, gr[sname] )

    # """
    # Testing the getMaterials() method of the Mili class.
    # """
    # def test_materials_getter(self):
    #     result = self.mili.labels_of_material(2)

    #     goal = [ {2: {'brick': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int32)}},
    #              {2: {'brick': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2], dtype=np.int32)}, 2: {'brick': np.array([1, 2, 3], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3], dtype=np.int32)}, 5: {'cseg': np.array([4, 5, 6], dtype=np.int32)}},
    #              {2: {'brick': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}, 5: {'cseg': np.array([7, 8, 9, 10, 11, 12], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2], dtype=np.int32)}, 2: {'brick': np.array([1, 2, 3], dtype=np.int32)}, 3: {'shell': np.array([1, 2, 3], dtype=np.int32)}, 4: {'cseg': np.array([1, 2, 3], dtype=np.int32)}, 5: {'cseg': np.array([4, 5, 6], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}},
    #              {1: {'beam': np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], dtype=np.int32)}} ]

    #     for pr, gr in zip(result,goal):
    #         for mat_num, mat in pr.items():
    #             self.assertTrue( mat_num in gr.keys() )
    #             for mat_name, labels, in mat.items():
    #                 self.assertTrue( mat_name in gr[mat_num].keys() )
    #                 np.testing.assert_equal( labels, gr[mat_num][mat_name])

    def test_reload_state_maps(self):
        """
        Test the reload_state_maps method of the Mili Class.
        """
        # Just make sure it doesn't cause Exception
        self.mili.reload_state_maps()
        state_maps = self.mili.state_maps()
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8
        self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
        for state_map in state_maps:
            self.assertEqual(len(state_map), STATE_COUNT)
            self.assertEqual(state_map[0].time, FIRST_STATE)
            self.assertEqual(state_map[-1].time, LAST_STATE)

    def test_material_numbers(self):
        mat_nums = self.mili.material_numbers()
        self.assertEqual(mat_nums[0], [2])
        self.assertEqual(mat_nums[1], [2])
        self.assertEqual(mat_nums[2], [1,2,3,4,5])
        self.assertEqual(mat_nums[3], [2,3,4,5])
        self.assertEqual(mat_nums[4], [1])
        self.assertEqual(mat_nums[5], [1,2,3,4,5])
        self.assertEqual(mat_nums[6], [1])
        self.assertEqual(mat_nums[7], [1])

    def test_classes_of_state_variable(self):
        sx_classes = self.mili.classes_of_state_variable('sx')
        self.assertEqual(sx_classes[0], ["brick"])
        self.assertEqual(sx_classes[1], ["brick"])
        self.assertEqual(set(sx_classes[2]), set(["beam", "shell", "brick"]))
        self.assertEqual(set(sx_classes[3]), set(["shell", "brick"]))
        self.assertEqual(sx_classes[4], ["beam"])
        self.assertEqual(set(sx_classes[5]), set(["beam", "shell", "brick"]))
        self.assertEqual(sx_classes[6], ["beam"])
        self.assertEqual(sx_classes[7], ["beam"])

        uy_classes = self.mili.classes_of_state_variable('uy')
        self.assertEqual(uy_classes[0], ["node"])
        self.assertEqual(uy_classes[1], ["node"])
        self.assertEqual(uy_classes[2], ["node"])
        self.assertEqual(uy_classes[3], ["node"])
        self.assertEqual(uy_classes[4], ["node"])
        self.assertEqual(uy_classes[5], ["node"])
        self.assertEqual(uy_classes[6], ["node"])
        self.assertEqual(uy_classes[7], ["node"])

        axf_classes = self.mili.classes_of_state_variable('axf')
        self.assertEqual(axf_classes[0], [])
        self.assertEqual(axf_classes[1], [])
        self.assertEqual(axf_classes[2], ["beam"])
        self.assertEqual(axf_classes[3], [])
        self.assertEqual(axf_classes[4], ["beam"])
        self.assertEqual(axf_classes[5], ["beam"])
        self.assertEqual(axf_classes[6], ["beam"])
        self.assertEqual(axf_classes[7], ["beam"])

    def test_materials_of_class_name(self):
        """
        Test the materials_of_class_name method of Mili Class.
        """
        brick_mats = self.mili.materials_of_class_name("brick")
        beam_mats = self.mili.materials_of_class_name("beam")
        shell_mats = self.mili.materials_of_class_name("shell")
        cseg_mats = self.mili.materials_of_class_name("cseg")

        self.assertEqual( brick_mats[0].size, 11 )
        self.assertEqual( brick_mats[1].size, 13 )
        self.assertEqual( brick_mats[2].size, 3 )
        self.assertEqual( brick_mats[3].size, 6 )
        self.assertEqual( brick_mats[4].size, 0 )
        self.assertEqual( brick_mats[5].size, 3 )
        self.assertEqual( brick_mats[6].size, 0 )
        self.assertEqual( brick_mats[7].size, 0 )
        np.testing.assert_equal( np.unique(brick_mats[0]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[1]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[2]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[3]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_mats[5]), np.array([2]) )

        self.assertEqual( beam_mats[0].size, 0 )
        self.assertEqual( beam_mats[1].size, 0 )
        self.assertEqual( beam_mats[2].size, 2 )
        self.assertEqual( beam_mats[3].size, 0 )
        self.assertEqual( beam_mats[4].size, 14 )
        self.assertEqual( beam_mats[5].size, 2 )
        self.assertEqual( beam_mats[6].size, 14 )
        self.assertEqual( beam_mats[7].size, 14 )
        np.testing.assert_equal( np.unique(beam_mats[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[4]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[5]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[6]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_mats[7]), np.array([1]) )

        self.assertEqual( shell_mats[0].size, 0 )
        self.assertEqual( shell_mats[1].size, 0 )
        self.assertEqual( shell_mats[2].size, 3 )
        self.assertEqual( shell_mats[3].size, 6 )
        self.assertEqual( shell_mats[4].size, 0 )
        self.assertEqual( shell_mats[5].size, 3 )
        self.assertEqual( shell_mats[6].size, 0 )
        self.assertEqual( shell_mats[7].size, 0 )
        np.testing.assert_equal( np.unique(shell_mats[2]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_mats[3]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_mats[5]), np.array([3]) )

        self.assertEqual( cseg_mats[0].size, 0 )
        self.assertEqual( cseg_mats[1].size, 0 )
        self.assertEqual( cseg_mats[2].size, 6 )
        self.assertEqual( cseg_mats[3].size, 12 )
        self.assertEqual( cseg_mats[4].size, 0 )
        self.assertEqual( cseg_mats[5].size, 6 )
        self.assertEqual( cseg_mats[6].size, 0 )
        self.assertEqual( cseg_mats[7].size, 0 )
        np.testing.assert_equal( cseg_mats[2], np.array([4, 4, 4, 5, 5, 5]) )
        np.testing.assert_equal( cseg_mats[3], np.array([4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]) )
        np.testing.assert_equal( cseg_mats[5], np.array([4, 4, 4, 5, 5, 5]))


    def test_parts_of_class_name(self):
        """
        Test the parts_of_class_name method of Mili Class.
        """
        brick_parts = self.mili.parts_of_class_name("brick")
        beam_parts = self.mili.parts_of_class_name("beam")
        shell_parts = self.mili.parts_of_class_name("shell")
        cseg_parts = self.mili.parts_of_class_name("cseg")

        self.assertEqual( brick_parts[0].size, 11 )
        self.assertEqual( brick_parts[1].size, 13 )
        self.assertEqual( brick_parts[2].size, 3 )
        self.assertEqual( brick_parts[3].size, 6 )
        self.assertEqual( brick_parts[4].size, 0 )
        self.assertEqual( brick_parts[5].size, 3 )
        self.assertEqual( brick_parts[6].size, 0 )
        self.assertEqual( brick_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(brick_parts[0]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[1]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[2]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[3]), np.array([2]) )
        np.testing.assert_equal( np.unique(brick_parts[5]), np.array([2]) )

        self.assertEqual( beam_parts[0].size, 0 )
        self.assertEqual( beam_parts[1].size, 0 )
        self.assertEqual( beam_parts[2].size, 2 )
        self.assertEqual( beam_parts[3].size, 0 )
        self.assertEqual( beam_parts[4].size, 14 )
        self.assertEqual( beam_parts[5].size, 2 )
        self.assertEqual( beam_parts[6].size, 14 )
        self.assertEqual( beam_parts[7].size, 14 )
        np.testing.assert_equal( np.unique(beam_parts[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[4]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[5]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[6]), np.array([1]) )
        np.testing.assert_equal( np.unique(beam_parts[7]), np.array([1]) )

        self.assertEqual( shell_parts[0].size, 0 )
        self.assertEqual( shell_parts[1].size, 0 )
        self.assertEqual( shell_parts[2].size, 3 )
        self.assertEqual( shell_parts[3].size, 6 )
        self.assertEqual( shell_parts[4].size, 0 )
        self.assertEqual( shell_parts[5].size, 3 )
        self.assertEqual( shell_parts[6].size, 0 )
        self.assertEqual( shell_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(shell_parts[2]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_parts[3]), np.array([3]) )
        np.testing.assert_equal( np.unique(shell_parts[5]), np.array([3]) )

        self.assertEqual( cseg_parts[0].size, 0 )
        self.assertEqual( cseg_parts[1].size, 0 )
        self.assertEqual( cseg_parts[2].size, 6 )
        self.assertEqual( cseg_parts[3].size, 12 )
        self.assertEqual( cseg_parts[4].size, 0 )
        self.assertEqual( cseg_parts[5].size, 6 )
        self.assertEqual( cseg_parts[6].size, 0 )
        self.assertEqual( cseg_parts[7].size, 0 )
        np.testing.assert_equal( np.unique(cseg_parts[2]), np.array([1]) )
        np.testing.assert_equal( np.unique(cseg_parts[3]), np.array([1]) )
        np.testing.assert_equal( np.unique(cseg_parts[5]), np.array([1]) )


    def test_mesh_object_classes_getter(self):
        """
        Test the mesh_object_classes() method of Mili Class.
        """
        MO_classes = self.mili.mesh_object_classes()

        # Test 0th processor
        mo_classes = MO_classes[0]

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 35)
        self.assertEqual(node_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 11)
        self.assertEqual(brick_class.idents_exist, True)

        # Test processor 5
        mo_classes = MO_classes[5]

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 27)
        self.assertEqual(node_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 3)
        self.assertEqual(brick_class.idents_exist, True)

        # shell class
        shell_class = mo_classes["shell"]
        self.assertEqual(shell_class.mesh_id, 0)
        self.assertEqual(shell_class.short_name, "shell")
        self.assertEqual(shell_class.long_name, "Shells")
        self.assertEqual(shell_class.sclass, Superclass.M_QUAD)
        self.assertEqual(shell_class.elem_qty, 3)
        self.assertEqual(shell_class.idents_exist, True)

        # cseg class
        cseg_class = mo_classes["cseg"]
        self.assertEqual(cseg_class.mesh_id, 0)
        self.assertEqual(cseg_class.short_name, "cseg")
        self.assertEqual(cseg_class.long_name, "Contact Segment")
        self.assertEqual(cseg_class.sclass, Superclass.M_QUAD)
        self.assertEqual(cseg_class.elem_qty, 6)

    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        result = self.mili.state_variables()
        goal = [159,126,134,134,126,134,126,126]
        result = [len(rr) for rr in result]
        self.assertEqual( result, goal )

    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.all_labels_of_material('es_13')
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

       answer = self.mili.all_labels_of_material('3')
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

       answer = self.mili.all_labels_of_material(3)
       np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
       np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
       np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))


    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1')
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material('1')
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material(1)
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elems('brick', 1)
        np.testing.assert_equal( answer[3][0][0,:], np.array([65,81,85,69,66,82,86,70],dtype=np.int32) )


    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query( 'matcgx', 'mat', labels = [1,2], states = 3 )
        np.testing.assert_equal( answer[0]['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32 ) )

    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query( 'nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['nodpos[ux]']['data'][0,0,0], 0.4330127537250519)
        answer = self.mili.query( 'ux', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['ux']['data'][0,0,0], 0.4330127537250519)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal(answer[3]['nodpos']['data'][0,0,:], np.array( [0.4330127537250519, 0.2500000596046448, 2.436666965484619], dtype = np.float32 ) )

    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )

        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )


    '''
    Testing the modification of a scalar state variable
    '''
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

    '''
    Testing the modification of a vector state variable
    '''
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

    '''
    Testing the modification of a vector component
    '''
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

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer[2]['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307] ], dtype = np.float32 ) )

    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        self.assertEqual(answer[2]['stress[sy]']['data'][0,0,0], -5373.53173828125)

    '''
    Test modifying a vector array
    '''
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
    '''
    Test modifying a vector array component
    '''
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

    def test_query_glob_results(self):
        """
        Test querying for results for M_MESH ("glob") element class.
        """
        answer = self.mili.query("he", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

        answer = self.mili.query("bve", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

        answer = self.mili.query("te", "glob", states=[22])
        self.assertAlmostEqual( answer[0]["te"]["data"][0,0,0], 1629.718, delta=1e-4)


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

class TestModifyUncombinedDatabase(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=True )

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
        self.mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=True )

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

class TestReturnCodes(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def test_loopwrapper(self):
        mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=True )
        # All procs return ReturnCode.ERROR so Exception is raised
        with self.assertRaises(ValueError):
            res = mili.query("does-not-exist", "brick")
        # Test some procs returning ReturnCode.ERROR so call succeeds
        res = mili.query("s1", "cseg", states = [1] )
        ret_codes = mili.returncode()
        self.assertEqual(ret_codes[0][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[1][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[2][0], ReturnCode.OK)
        self.assertEqual(ret_codes[3][0], ReturnCode.OK)
        self.assertEqual(ret_codes[4][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[5][0], ReturnCode.OK)
        self.assertEqual(ret_codes[6][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[7][0], ReturnCode.ERROR)

    def test_poolwrapper(self):
        mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=False )
        # All procs return ReturnCode.ERROR so Exception is raised
        with self.assertRaises(ValueError):
            res = mili.query("does-not-exist", "brick")
        # Test some procs returning ReturnCode.ERROR so call succeeds
        res = mili.query("s1", "cseg", states = [1] )
        # We don't check return code's here because they are not maintained
        # across calls to the pool wrapper. They are all handled internally.
        # This is fine because we don't expect users to access the returncode methd.

    def test_serverwrapper(self):
        mili = reader.open_database( TestCombineFunction.file_name, suppress_parallel=False, experimental=True )
        # All procs return ReturnCode.ERROR so Exception is raised
        with self.assertRaises(ValueError):
            res = mili.query("does-not-exist", "brick")
        # Test some procs returning ReturnCode.ERROR so call succeeds
        res = mili.query("s1", "cseg", states = [1] )
        ret_codes = mili.returncode()
        self.assertEqual(ret_codes[0][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[1][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[2][0], ReturnCode.OK)
        self.assertEqual(ret_codes[3][0], ReturnCode.OK)
        self.assertEqual(ret_codes[4][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[5][0], ReturnCode.OK)
        self.assertEqual(ret_codes[6][0], ReturnCode.ERROR)
        self.assertEqual(ret_codes[7][0], ReturnCode.ERROR)

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

    def test_serial_append_single_state_zero_out_false(self):
        """Test appending a single state to an existing database. zero_out=False"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name )
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

    def test_serial_append_multiple_states_zero_out_false(self):
        """Test appending multiple states in a row to an existing database. zero_out is False"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name )
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

    def test_serial_append_single_state_zero_out_true(self):
        """Test appending a single state to an existing database. zero_out=True"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name )
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

    def test_serial_append_multiple_states_zero_out_true(self):
        """Test appending multiple states in a row to an existing database. zero_out is True"""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name )
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

    def test_serial_append_bad_time(self):
        """Test appending an invalid state time."""
        db_name = f"{self._testMethodName}_{TestAppendStateSerial.base_name}"
        mili = reader.open_database( db_name )
        with self.assertRaises(ValueError):
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

    def test_parallel_append_single_state_zero_out_false(self):
        """Test appending a single state to an existing database. zero_out=False"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, experimental=True )
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

    def test_parallel_append_multiple_states_zero_out_false(self):
        """Test appending multiple states in a row to an existing database. zero_out is False"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, experimental=True )
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

    def test_parallel_append_single_state_zero_out_true(self):
        """Test appending a single state to an existing database. zero_out=True"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, experimental=True )
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

    def test_parallel_append_multiple_states_zero_out_true(self):
        """Test appending multiple states in a row to an existing database. zero_out is True"""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, experimental=True )
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

    def test_parallel_append_bad_time(self):
        """Test appending an invalid state time."""
        db_name = f"{self._testMethodName}_{TestAppendStateParallel.base_name}"
        mili = reader.open_database( db_name, suppress_parallel=False, experimental=True )
        with self.assertRaises(ValueError):
            mili.append_state( 0.00 )

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
                                        TestCopyNonStateDataSerial.base_name), suppress_parallel=True )

    def tearDown(self):
        if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}A"):
            os.remove(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}A")
        if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}00"):
            os.remove(f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}00")

    def test_copy(self):
        """Tests creating a copy of an existing database without the state data."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name, suppress_parallel=True )

        # Check that state maps, times and state counts are all 0
        self.assertEqual( len(new_db.state_maps()), 0 )
        self.assertEqual( len(new_db.times()), 0 )
        self.assertEqual( new_db.parameter('state_count'), 0 )

        # Check that everything else looks the same
        np.testing.assert_equal( new_db.nodes(), self.db.nodes() )
        self.dictionary_diff_helper( new_db.labels(), self.db.labels() )
        self.dictionary_diff_helper( new_db.connectivity(), self.db.connectivity() )
        self.assertEqual( new_db.mesh_dimensions(), self.db.mesh_dimensions() )
        self.assertEqual( new_db.element_sets(), self.db.element_sets() )
        self.dictionary_diff_helper(self.db.state_variables(), new_db.state_variables())
        self.assertEqual(self.db.subrecords(), new_db.subrecords())

    def test_copy_append_state(self):
        """Tests copying existing database and then appending a state."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataSerial.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name, suppress_parallel=True )
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
                                        suppress_parallel=False, experimental=True )

    def tearDown(self):
        for proc in ["000", "001", "002", "003", "004", "005", "006", "007"]:
            if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}A"):
                os.remove(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}A")
            if os.path.exists(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}00"):
                os.remove(f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}{proc}00")

    def test_copy(self):
        """Tests creating a copy of an existing database without the state data."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, experimental=True )

        # Check that state maps, times and state counts are all 0 for all processors
        state_maps = new_db.state_maps()
        state_times = new_db.times()
        self.assertEqual(len(state_maps), 8)
        for processor_smap, processor_times in zip(state_maps, state_times):
            self.assertEqual( len(processor_smap), 0 )
            self.assertEqual( len(processor_times), 0 )
        self.assertEqual( new_db.parameter('state_count'), [0,0,0,0,0,0,0,0] )

        # Check that everything else looks the same
        np.testing.assert_equal( new_db.nodes(), self.db.nodes() )
        for orig_labels, new_labels in zip(self.db.labels(), new_db.labels() ):
            self.dictionary_diff_helper( orig_labels, new_labels )
        for orig_conns, new_conns in zip(self.db.connectivity(), new_db.connectivity() ):
            self.dictionary_diff_helper( orig_conns, new_conns )
        self.assertEqual( new_db.mesh_dimensions(), self.db.mesh_dimensions() )
        self.assertEqual( new_db.element_sets(), self.db.element_sets() )
        self.assertEqual( new_db.state_variables(), self.db.state_variables() )
        self.assertEqual( new_db.subrecords(), self.db.subrecords() )

    def test_copy_append_state(self):
        """Tests copying existing database and then appending a state."""
        new_db_name = f"{self._testMethodName}_{TestCopyNonStateDataParallel.new_base_name}"
        self.db.copy_non_state_data( new_db_name )

        new_db = reader.open_database( new_db_name,
                                       suppress_parallel=False, experimental=True )
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

if __name__ == "__main__":
    unittest.main()
