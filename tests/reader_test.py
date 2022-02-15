#!/usr/bin/env python3

"""
Copyright (c) 2016-2021, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by 
 William Tobin (tobin6@llnl.hov) and Kevin Durrenberger (durrenberger1@llnl.gov). 
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
from mili import reader
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

'''
These are tests to assert the correctness of the Mili Reader
These tests use d3samp6.plt
'''
class TestMiliReader(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','d3samp6.plt')
    '''
    Set up the mili file object
    '''
    def setUp(self):
        self.mili = reader.open_database( TestMiliReader.file_name, suppress_parallel = True )

    def test_serial_read(self):
        reader.open_database( TestMiliReader.file_name, suppress_parallel = True )

    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError) as context:
            self.mili.all_labels_of_material('es_13121')
        with self.assertRaises(ValueError) as context:
            self.mili.nodes_of_material('es_13121')
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[uxxx]'], 'node', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodposss[ux]'], 'node', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', material = 'cat', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(4, 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
        with self.assertRaises(TypeError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
        with self.assertRaises(TypeError) as context:
            self.mili.query(['nodposss[ux]'], 'node', labels = 4, states = 3, ips = 'cat')

    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
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


    """
    Testing the getStateMaps() method of the Mili class
    """
    def test_statemaps_getter(self):
        FIRST_STATE = 0.0 
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101

        state_maps = self.mili.state_maps()
        num_state_maps = len(state_maps)
        
        self.assertEqual(STATE_COUNT, num_state_maps)
        self.assertEqual(FIRST_STATE, state_maps[0].time)
        self.assertEqual(LAST_STATE, state_maps[num_state_maps-1].time)


    """
    Testing the getLabels() method of the Mili class
    """
    def test_labels_getter(self):
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


    # """
    # Testing the getMaterials() method of the Mili class.
    # """
    # def test_materials_getter(self):
    #     materials = self.mili.materials()
    #     MATERIALS = { 1: {'beam': np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
    #                                19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    #                                35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46] ) },
    #                   2: {'brick': np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
    #                                 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
    #                                 35, 36] ) },
    #                   3: {'shell': np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ) },
    #                   4: {'cseg': np.array( [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] ) },
    #                   5: {'cseg': np.array( [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24] ) } 
    #                 }
    #     for k1, k2 in zip( materials.keys(), MATERIALS.keys() ):
    #         self.assertEqual( k1, k2 )
    #         for m1, m2 in zip( materials[k1].keys(), materials[k2].keys() ):
    #             self.assertEqual( m1, m2 )
    #             ml1 = materials[k1][m1]
    #             ml2 = materials[k2][m2]
    #             np.testing.assert_equal( ml1, ml2 )


    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        state_variable_names = list(self.mili.state_variables().keys())
        SVAR_NAMES = ['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
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
                      'dam', 'frac_strain', 'sand', 'cause'] 
        self.assertEqual(state_variable_names, SVAR_NAMES)


    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.all_labels_of_material('es_13')
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
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elems('brick', 1)
        self.assertEqual(answer.size, 8)
        np.testing.assert_equal(answer, np.array( [[65, 81, 85, 69, 66, 82, 86, 70]], dtype = np.int32 ))
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query('matcgx', 'mat', labels = [1,2], states = 3 )
        self.assertEqual(answer['matcgx']['layout']['states'][0], 3)
        self.assertEqual(list(answer.keys()), ['matcgx'] )
        np.testing.assert_equal( answer['matcgx']['layout']['mat'], np.array( [ 1, 2 ], dtype = np.int32) )
        np.testing.assert_equal( answer['matcgx']['data']['mat'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32) )

    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query('nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual(answer['nodpos[ux]']['layout']['node'][0], 70)
        self.assertEqual(answer['nodpos[ux]']['data']['node'][0], 0.4330127537250519 )
        
        answer = self.mili.query('ux', 'node', labels = 70, states = 3 )
        self.assertEqual(answer['ux']['layout']['node'][0], 70)
        self.assertEqual(answer['ux']['data']['node'][0], 0.4330127537250519)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        self.assertEqual(answer['sx']['layout']['2hex_mmsvn_rec'].size, 36)
        
        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        self.assertEqual(answer['sx']['layout']['2hex_mmsvn_rec'].size, 36)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,:,:], np.array( [ [ 0.4330127537250519, 0.2500000596046448, 2.436666965484619 ] ], dtype = np.float32 ) )
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        v1 = { 'matcgx' : 
                { 'layout' : 
                    { 
                      'mat' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    }, 
                  'data' : { 'mat' : np.array([ [ [ 5.5 ] ] ], dtype = np.float32) } 
                } 
             }
        v2 = { 'matcgx' : 
                { 'layout' : 
                    { 
                      'mat' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    }, 
                  'data' : 
                    { 'mat' : np.array([ [ [ 0.6021666526794434 ] ] ], dtype = np.float32) }
                } 
             }

        # Original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3 )
        self.assertEqual( answer['matcgx']['data']['mat'][0], 0.6021666526794434 )

        # Modified
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v1 )
        self.assertEqual( answer['matcgx']['data']['mat'][0], 5.5 )

        # Back to original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v2 )
        self.assertEqual( answer['matcgx']['data']['mat'][0], 0.6021666526794434 )
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        v1 = { 'nodpos' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 ) 
                    }, 
                  'data' : 
                    { 'node' : np.array( [ [ [ 5.0, 6.0, 9.0 ], [ 5.1, 6.1, 9.1 ] ] ], dtype = np.float32 ) } 
                } 
             }
        v2 = { 'nodpos' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )  
                    }, 
                  'data' : 
                    { 'node' : np.array( [ [ [0.4330127537250519, 0.2500000596046448, 2.436666965484619], [0.4330127239227295, 0.2499999850988388, 2.7033333778381348] ] ], dtype = np.float32 ) } 
                } 
             }

        # Before change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4 )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,0,:], v2['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,1,:], v2['nodpos']['data']['node'][0,1,:] )

        # After change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4, write_data = v1 )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,0,:], v1['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,1,:], v1['nodpos']['data']['node'][0,1,:] )
        
        # Back to original
        answer = self.mili.query('nodpos', 'node', labels = [70, 71],  states = 4, write_data = v2 )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,0,:], v2['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer['nodpos']['data']['node'][0,1,:], v2['nodpos']['data']['node'][0,1,:] )
    
    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        v1 = { 'nodpos[uz]' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : 
                    { 'node' : np.array( [ [ [9.0], [9.0] ] ], dtype = np.float32 ) } 
                } 
             }
        v2 = { 'nodpos[uz]' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : 
                    { 'node' : np.array( [ [ [2.436666965484619], [2.7033333778381348] ] ], dtype = np.float32 ) } 
                } 
             }
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4 )
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,0,0], 2.436666965484619)
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,1,0], 2.7033333778381348)

        # After change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v1 )
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,0,0], 9.0)
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,1,0], 9.0)

        # Back to original
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v2 )
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,0,0], 2.436666965484619)
        self.assertEqual(answer['nodpos[uz]']['data']['node'][0,1,0], 2.7033333778381348)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer['stress']['data']['1beam_mmsvn_rec'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307 ] ], dtype = np.float32 ) )
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        self.assertEqual(answer['stress[sy]']['data']['1beam_mmsvn_rec'][0,0,0], -5545.70751953125)
        
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
        v1 = { 'stress' : 
                { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                  'data' : 
                    { '1beam_mmsvn_rec' : np.array([ [ [ -5547.1025390625, -5545.70751953125, -3.736035978363361e-07, 5546.4052734375, 0.4126972556114197, -0.412697434425354 ] ] ], dtype = np.float32 ) } 
                } 
             }
        v2 = { 'stress' : 
                { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                  'data' : 
                    { '1beam_mmsvn_rec' : np.array([ [ [ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 ] ] ], dtype = np.float32 ) } 
                } 
             }
        # Before change
        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2 )
        np.testing.assert_equal( answer['stress']['data']['1beam_mmsvn_rec'], v1['stress']['data']['1beam_mmsvn_rec'] )

        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress']['data']['1beam_mmsvn_rec'], v2['stress']['data']['1beam_mmsvn_rec'] )
        
        # Back to original
        answer = self.mili.query('stress', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress']['data']['1beam_mmsvn_rec'], v1['stress']['data']['1beam_mmsvn_rec'] )
    
    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        v1 = { 'stress[sy]' :
                 { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                   'data' : 
                    { '1beam_mmsvn_rec' : np.array( [ [ [ -5545.70751953125 ] ] ], dtype = np.float32 ) }
                 }
             }
        v2 = { 'stress[sy]' :
                 { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [71], dtype = np.int32 )
                    },
                   'data' : 
                    { '1beam_mmsvn_rec' : np.array( [ [ [ 12.0 ] ] ], dtype = np.float32 ) }
                 }
             }

        # Before change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        np.testing.assert_equal( answer['stress[sy]']['data']['1beam_mmsvn_rec'], v1['stress[sy]']['data']['1beam_mmsvn_rec'] )
        
        # After change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer['stress[sy]']['data']['1beam_mmsvn_rec'], v2['stress[sy]']['data']['1beam_mmsvn_rec'] )
        
        # Back to original
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer['stress[sy]']['data']['1beam_mmsvn_rec'], v1['stress[sy]']['data']['1beam_mmsvn_rec'] )
    
'''
Testing the parallel Mili file version
'''
class TestMiliReaderParallel(unittest.TestCase):
    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6.plt')
    '''
    Set up the mili file object
    '''
    def setUp(self):
        self.mili = reader.open_database( TestMiliReaderParallel.file_name )
    
    def test_parallel_read(self):
        reader.open_database( TestMiliReaderParallel.file_name )

    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError) as context:
            self.mili.all_labels_of_material('es_13121')
        with self.assertRaises(ValueError) as context:
            self.mili.nodes_of_material('es_13121')
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[uxxx]'], 'node', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodposss[ux]'], 'node', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', material = 'cat', labels = 4, states = 3)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(4, 'node', labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
        with self.assertRaises(ValueError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
        with self.assertRaises(TypeError) as context:
            self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
        with self.assertRaises(TypeError) as context:
            self.mili.query(['nodposss[ux]'], 'node', labels = 4, states = 3, ips = 'cat')


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


    """
    Testing the getStateMaps() method of the Mili class
    """
    def test_statemaps_getter(self):
        state_maps = self.mili.state_maps()
        FIRST_STATE = 0.0 
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        PROCS = 8

        self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
        for i in range(PROCS):
            procs_state_maps = state_maps[i]
            self.assertEqual(len(procs_state_maps), STATE_COUNT)
            self.assertEqual(procs_state_maps[0].time, FIRST_STATE)
            self.assertEqual(procs_state_maps[STATE_COUNT-1].time, LAST_STATE)


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

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1')
        node_labels = np.unique( np.concatenate((*answer,)) )
        np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elems('brick', 1)
        np.testing.assert_equal( answer[3][0,:], np.array([65,81,85,69,66,82,86,70],dtype=np.int32) )

    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query( 'matcgx', 'mat', labels = [1,2], states = 3 )
        np.testing.assert_equal( answer[0]['matcgx']['data']['mat'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32 ) )
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query( 'nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['nodpos[ux]']['data']['node'][0,0,0], 0.4330127537250519)
        answer = self.mili.query( 'ux', 'node', labels = 70, states = 3 )
        self.assertEqual( answer[3]['ux']['data']['node'][0,0,0], 0.4330127537250519)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal(answer[3]['nodpos']['data']['node'][0,0,:], np.array( [0.4330127537250519, 0.2500000596046448, 2.436666965484619], dtype = np.float32 ) )
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( '2hex_mmsvn_rec', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )
        
        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        num_labels = sum( pansw['sx']['layout'].get( '2hex_mmsvn_rec', np.empty([0],dtype=np.int32) ).size for pansw in answer )
        self.assertEqual( num_labels, 36 )
    
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        v1 = { 'matcgx' : 
                { 'layout' : 
                    { 
                      'mat' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    }, 
                  'data' : { 'mat' : np.array([ [ [ 5.5 ] ] ], dtype = np.float32) } 
                } 
             }
        v2 = { 'matcgx' : 
                { 'layout' : 
                    { 
                      'mat' : np.array( [1], dtype = np.int32 ),
                      'states' : np.array( [3], dtype = np.int32 )
                    }, 
                  'data' : 
                    { 'mat' : np.array([ [ [ 0.6021666526794434 ] ] ], dtype = np.float32) }
                } 
             }
    
        # Original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3 )
        self.assertEqual( answer[0]['matcgx']['data']['mat'][0], 0.6021666526794434 )

        # Modified
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v1 )
        self.assertEqual( answer[0]['matcgx']['data']['mat'][0], 5.5 )

        # Back to original
        answer = self.mili.query('matcgx', 'mat', labels = 1, states = 3, write_data = v2 )
        self.assertEqual( answer[0]['matcgx']['data']['mat'][0], 0.6021666526794434 )
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        v1 = { 'nodpos' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 ) 
                    }, 
                  'data' : 
                    { 'node' : np.array( [ [ [ 5.0, 6.0, 9.0 ], [ 5.1, 6.1, 9.1 ] ] ], dtype = np.float32 ) } 
                }
             }
        v2 = { 'nodpos' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [ 70, 71 ], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )  
                    }, 
                  'data' : 
                    { 'node' : np.array( [ [ [0.4330127537250519, 0.2500000596046448, 2.436666965484619], [0.4330127239227295, 0.2499999850988388, 2.7033333778381348] ] ], dtype = np.float32 ) } 
                }
             }

        # Before change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4 )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,0,:], v2['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,1,:], v2['nodpos']['data']['node'][0,1,:] )

        # After change
        answer = self.mili.query('nodpos', 'node', labels = [70, 71], states = 4, write_data = v1 )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,0,:], v1['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,1,:], v1['nodpos']['data']['node'][0,1,:] )

        # Back to original
        answer = self.mili.query('nodpos', 'node', labels = [70, 71],  states = 4, write_data = v2 )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,0,:], v2['nodpos']['data']['node'][0,0,:] )
        np.testing.assert_equal( answer[0]['nodpos']['data']['node'][0,1,:], v2['nodpos']['data']['node'][0,1,:] )

    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        v1 = { 'nodpos[uz]' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : 
                    { 'node' : np.array( [ [ [9.0], [9.0] ] ], dtype = np.float32 ) } 
                } 
             }
        v2 = { 'nodpos[uz]' : 
                { 'layout' : 
                    { 
                        'node' : np.array( [70, 71], dtype = np.int32 ),
                        'states' : np.array( [4], dtype = np.int32 )
                    },
                  'data' : 
                    { 'node' : np.array( [ [ [2.436666965484619], [2.7033333778381348] ] ], dtype = np.float32 ) } 
                } 
             }
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4 )
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,0,0], 2.436666965484619)
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,1,0], 2.7033333778381348)

        # After change
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v1 )
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,0,0], 9.0)
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,1,0], 9.0)

        # Back to original
        answer = self.mili.query('nodpos[uz]', 'node', labels = [70, 71], states = 4, write_data = v2 )
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,0,0], 2.436666965484619)
        self.assertEqual(answer[0]['nodpos[uz]']['data']['node'][0,1,0], 2.7033333778381348)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer[2]['stress']['data']['1beam_mmsvn_rec'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307 ] ], dtype = np.float32 ) )

    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        self.assertEqual(answer[2]['stress[sy]']['data']['1beam_mmsvn_rec'][0,0,0], -5373.53173828125)
    
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
        v1 = { 'stress' : 
                { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                  'data' : 
                    { '1beam_mmsvn_rec' : np.array([ [ [ -5377.6376953125, -5373.53173828125, -3.930831553589087e-07, 5375.58447265625, 0.6931889057159424, -0.693189263343811 ] ] ], dtype = np.float32 ) } 
                } 
             }
        v2 = { 'stress' : 
                { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                  'data' : 
                    { '1beam_mmsvn_rec' : np.array([ [ [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ] ] ], dtype = np.float32 ) } 
                } 
             }
        # Before change
        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2 )
        np.testing.assert_equal( answer[2]['stress']['data']['1beam_mmsvn_rec'], v1['stress']['data']['1beam_mmsvn_rec'] )

        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer[2]['stress']['data']['1beam_mmsvn_rec'], v2['stress']['data']['1beam_mmsvn_rec'] )
        
        # Back to original
        answer = self.mili.query('stress', 'beam', labels = 5, states = 70, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer[2]['stress']['data']['1beam_mmsvn_rec'], v1['stress']['data']['1beam_mmsvn_rec'] )
    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        v1 = { 'stress[sy]' :
                 { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                   'data' : 
                    { '1beam_mmsvn_rec' : np.array( [ [ [ -5373.53173828125 ] ] ], dtype = np.float32 ) }
                 }
             }
        v2 = { 'stress[sy]' :
                 { 'layout' : 
                    { 
                        '1beam_mmsvn_rec' : np.array( [5], dtype = np.int32 ),
                        'states' : np.array( [70], dtype = np.int32 )
                    },
                   'data' : 
                    { '1beam_mmsvn_rec' : np.array( [ [ [ 1.5 ] ] ], dtype = np.float32 ) }
                 }
             }

        # Before change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data']['1beam_mmsvn_rec'], v1['stress[sy]']['data']['1beam_mmsvn_rec'] )
        
        # After change
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2, write_data = v2 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data']['1beam_mmsvn_rec'], v2['stress[sy]']['data']['1beam_mmsvn_rec'] )
        
        # Back to original
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2, write_data = v1 )
        np.testing.assert_equal( answer[2]['stress[sy]']['data']['1beam_mmsvn_rec'], v1['stress[sy]']['data']['1beam_mmsvn_rec'] )
