#!/usr/bin/env python3

"""
Copyright (c) 2016, Lawrence Livermore National Security, LLC. 
 Produced at the Lawrence Livermore National Laboratory. Written 
 by Kevin Durrenberger: durrenberger1@llnl.gov. CODE-OCEC-16-056. 
 All rights reserved.

 This file is part of Mili. For details, see <URL describing code 
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
import unittest
import mili_reader_lib as reader
import os, sys
import psutil
'''
These are tests to assert the correctness of the Mili Reader

These tests use d3samp6.plt
'''
class TestMiliReader(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):        
        file_name = 'd3samp6.plt'
        #file_name = 'states/d3samp6.plt'    
        self.mili = reader.Mili(file_name)
        self.mili.setErrorFile()
        
    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        tests = [ self.mili.labels_of_material('es_13121'),
                  self.mili.nodes_of_material('es_13121'),
                  self.mili.nodes_of_elem('1', 'bsseam'),
                  self.mili.nodes_of_elem('-1', 'beam'),
                  self.mili.query(['nodpos[uxxx]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'nodels', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', material='cat', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=-4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 4, labels=4, state_numbers=300),
                  self.mili.query([4], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', material=9, labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels='cat', state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers='cat'),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=3, modify='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, int_points='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, raw_data='cat') ]
        assert( not any( tests ) )


    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
        NUM_NODES = 144
        FIRST_NODE = [(1.0, 0.0, 0.0)]
        LAST_NODE = [(-6.556708598282057e-08, 1.5, 3.0)]
        NODE9 = [(0.5049999356269836, 0.8746857047080994, 1.0)]
        NODE20 = [(0.8677574396133423, 0.5009999871253967, 0.20000000298023224)]  
        NODE32 = [(1.0019999742507935, 0.0, 1.7999999523162842)] 
        NODE54 = [(0.3749999701976776, 0.6495190858840942, 2.0)] 
        NODE63 = [(-6.556708598282057e-08, 1.5, 2.0)]
        NODE88 = [(0.3749999701976776, 0.6495190858840942, 2.200000047683716)]
        NODE111 = [(-4.371138828673793e-08, 1.0, 3.0)]
        NODE124 = [(-5.463923713477925e-08, 1.25, 2.200000047683716)]

        nodes = self.mili.getNodes()
        num_nodes = len(nodes)

        self.assertEqual(num_nodes, NUM_NODES)
        self.assertEqual(nodes[0], FIRST_NODE)
        self.assertEqual(nodes[num_nodes-1], LAST_NODE)
        self.assertEqual(nodes[9], NODE9)
        self.assertEqual(nodes[20], NODE20)
        self.assertEqual(nodes[32], NODE32)
        self.assertEqual(nodes[54], NODE54)
        self.assertEqual(nodes[63], NODE63)
        self.assertEqual(nodes[88], NODE88)
        self.assertEqual(nodes[111], NODE111)
        self.assertEqual(nodes[124], NODE124)


    """
    Testing the getStateMaps() method of the Mili class
    """
    def test_statemaps_getter(self):
        FIRST_STATE = 0.0 
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101

        state_maps = self.mili.getStateMaps()
        num_state_maps = len(state_maps)
        
        self.assertEqual(STATE_COUNT, num_state_maps)
        self.assertEqual(FIRST_STATE, state_maps[0].time)
        self.assertEqual(LAST_STATE, state_maps[num_state_maps-1].time)


    """
    Testing the getLabels() method of the Mili class
    """
    def test_labels_getter(self):
        labels = self.mili.getLabels()

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

        NODE_LBLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                     13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 22: 22,
                     23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 31: 31, 32: 32,
                     33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 40: 40, 41: 41, 42: 42,
                     43: 43, 44: 44, 45: 45, 46: 46, 47: 47, 48: 48, 49: 49, 50: 50, 51: 51, 52: 52,
                     53: 53, 54: 54, 55: 55, 56: 56, 57: 57, 58: 58, 59: 59, 60: 60, 61: 61, 62: 62,
                     63: 63, 64: 64, 65: 65, 66: 66, 67: 67, 68: 68, 69: 69, 70: 70, 71: 71, 72: 72,
                     73: 73, 74: 74, 75: 75, 76: 76, 77: 77, 78: 78, 79: 79, 80: 80, 81: 81, 82: 82,
                     83: 83, 84: 84, 85: 85, 86: 86, 87: 87, 88: 88, 89: 89, 90: 90, 91: 91, 92: 92,
                     93: 93, 94: 94, 95: 95, 96: 96, 97: 97, 98: 98, 99: 99, 100: 100, 101: 101,
                     102: 102, 103: 103, 104: 104, 105: 105, 106: 106, 107: 107, 108: 108, 109: 109,
                     110: 110, 111: 111, 112: 112, 113: 113, 114: 114, 115: 115, 116: 116, 117: 117,
                     118: 118, 119: 119, 120: 120, 121: 121, 122: 122, 123: 123, 124: 124, 125: 125,
                     126: 126, 127: 127, 128: 128, 129: 129, 130: 130, 131: 131, 132: 132, 133: 133,
                     134: 134, 135: 135, 136: 136, 137: 137, 138: 138, 139: 139, 140: 140, 141: 141,
                     142: 142, 143: 143, 144: 144}  
        BEAM_LBLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                     13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21, 
                     22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 
                     31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36, 37: 37, 38: 38, 39: 39, 
                     40: 40, 41: 41, 42: 42, 43: 43, 44: 44, 45: 45, 46: 46} 
        BRICK_LBLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12,
                      13: 13, 14: 14, 15: 15, 16: 16, 17: 17, 18: 18, 19: 19, 20: 20, 21: 21,
                      22: 22, 23: 23, 24: 24, 25: 25, 26: 26, 27: 27, 28: 28, 29: 29, 30: 30, 
                      31: 31, 32: 32, 33: 33, 34: 34, 35: 35, 36: 36} 
        SHELL_LBLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12} 
        GLOB_LBLS = {1: 1}
        MATS_LBLS = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        self.assertEqual(labels['node'], NODE_LBLS)
        self.assertEqual(labels['beam'], BEAM_LBLS)
        self.assertEqual(labels['brick'], BRICK_LBLS)
        self.assertEqual(labels['shell'], SHELL_LBLS)
        self.assertEqual(labels['glob'], GLOB_LBLS)
        self.assertEqual(labels['mat'], MATS_LBLS)


    """
    Testing the getMaterials() method of the Mili class.
    """
    def test_materials_getter(self):
        materials = self.mili.getMaterials()
        
        MATERIALS =  {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                   19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                   35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46]},
                      2: {'brick': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                    19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                                    35, 36]},
                      3: {'shell': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
                      4: {'cseg': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
                      5: {'cseg': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]}}
        self.assertEqual(materials, MATERIALS)


    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        state_variable_names = list(self.mili.getStateVariables().keys())
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
       answer = self.mili.labels_of_material('es_13', False)
       self.assertEqual(len(answer.items), 12)
       for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].mo_id, i + 1)

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        self.assertEqual(len(answer.items), 46)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [65, 81, 85, 69, 66, 82, 86, 70]
        self.assertEqual(len(answer.items), 8)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matcgx'], 'mat', None, [1,2], [3], raw_data=False)
        self.assertEqual(len(answer.state_answers), 1)
        arr = [0.6021666526794434, 0.6706029176712036]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                self.assertEqual(answer.state_answers[i].items[j].value, arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
        
        answer = self.mili.query(['ux'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
        
        answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [70], [4], None, None, False)
        arr = [0.4330127537250519, 0.2500000596046448, 2.436666965484619]
        self.assertEqual(answer.state_answers[0].items[0].value, arr)
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        val = {3 : {'matcgx' : {1 : 5.5}}}
        val2 = {3 : {'matcgx' : {1 : 0.6021666526794434}}}
    
        # Before change
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.6021666526794434)
        
        # After change
        self.mili.modify_state_variable('matcgx', 'mat', val, 1, [3])
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 5.5)
        
        # Back to original
        self.mili.modify_state_variable('matcgx', 'mat', val2, 1, [3])
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.6021666526794434)
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        val = {4 : {'nodpos' : {70 : [5.0, 6.0, 9.0], 71: [5.0, 6.0, 9.0]}}}
        val2 = {4 : {'nodpos' : {70 : [0.4330127537250519, 0.2500000596046448, 2.436666965484619], 71: [0.4330127239227295, 0.2499999850988388, 2.7033333778381348]}}}
        
        # Before change
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [0.4330127537250519, 0.2500000596046448, 2.436666965484619])
        self.assertEqual(answer.state_answers[0].items[1].value, [0.4330127239227295, 0.2499999850988388, 2.7033333778381348])
        
        
        # After change
        self.mili.modify_state_variable('nodpos', 'node', val, [70, 71], [4])
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [5.0, 6.0, 9.0])
        self.assertEqual(answer.state_answers[0].items[1].value, [5.0, 6.0, 9.0])
        
        # Back to original
        self.mili.modify_state_variable('nodpos', 'node', val2, [70, 71], [4])
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [0.4330127537250519, 0.2500000596046448, 2.436666965484619])
        self.assertEqual(answer.state_answers[0].items[1].value, [0.4330127239227295, 0.2499999850988388, 2.7033333778381348])
    
    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        val = {4 : {'nodpos[uz]' : {70 : 9.0, 71: 9.0}}}
        val2 = {4 : {'nodpos[uz]' : {70 : 2.436666965484619, 71 : 2.7033333778381348}}}
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 2.436666965484619)
        self.assertEqual(answer.state_answers[0].items[1].value, 2.7033333778381348)
    
        
        # After change
        self.mili.modify_state_variable('nodpos[uz]', 'node', val, [70, 71], [4])
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 9.0)
        self.assertEqual(answer.state_answers[0].items[1].value, 9.0)
    
        
        # Back to original
        self.mili.modify_state_variable('nodpos[uz]', 'node', val2, [70, 71], [4])
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 2.436666965484619)
        self.assertEqual(answer.state_answers[0].items[1].value, 2.7033333778381348)
    
    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [21,22], False, [2], False)
        base_answer = {'sx': {2: -1018.4232177734375}, 'sy': {2: -1012.2537231445312}, 'sz': {2: -6.556616085617861e-07}, 'sxy': {2: 1015.3384399414062}, 'syz': {2: 0.3263571858406067}, 'szx': {2: -0.32636013627052307}}
        self.assertEqual(answer.state_answers[1].items[0].value, base_answer)
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [71], False, [2], False)
        base_answer = {'sy' : {2: -5545.70751953125}}
        self.assertEqual(answer.state_answers[0].items[0].value, base_answer)
        
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
         # Before change
        d = {71 : {'stress' : {5 : {'sz': {2: -3.736035978363361e-07}, 'sy': {2: -5545.70751953125}, 'sx': {2: -5547.1025390625}, 'szx': {2: -0.412697434425354}, 'sxy': {2: 5546.4052734375}, 'syz': {2: 0.4126972556114197}}}}}
        dd = {'sz': {2: -3.736035978363361e-07}, 'sy': {2: -5545.70751953125}, 'sx': {2: -5547.1025390625}, 'szx': {2: -0.412697434425354}, 'sxy': {2: 5546.4052734375}, 'syz': {2: 0.4126972556114197}}

        answer = self.mili.query('stress', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)
        
        mod = {71 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [71], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [71], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)
    
    '''
    Test modifying a vector array component
    '''
    # ANSWER UPDATED
    def test_modify_vector_array_component(self):
        # OLD ORIGINAL VALUE d = {71 : {'stress' : {5 : {'sy': {2: -0.40591922402381897}}}}}
        # OLD ANSWER dd = -0.40591922402381897
        d = {71 : {'stress' : {5 : {'sy': {2: -5545.70751953125}}}}}
        dd = -5545.70751953125
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)
        
        # After change
        mod = {71 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [71], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [71], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [71], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)
    
'''
Testing the parallel Mili file version
'''
class TestMiliReaderParallel(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
        file_name = 'parallel/d3samp6.plt'
        self.mili = reader.Mili()
        self.mili.read(file_name, parallel_read=False)
        self.mili.setErrorFile()   
    
    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        tests = [ self.mili.labels_of_material('es_13121'),
                  self.mili.nodes_of_material('es_13121'),
                  self.mili.nodes_of_elem('1', 'bsseam'),
                  self.mili.nodes_of_elem('-1', 'beam'),
                  self.mili.query(['nodpos[uxxx]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'nodels', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', material='cat', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=-4,state_numbers= 3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 4, labels=4, state_numbers=300),
                  self.mili.query([4], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', 9, labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels='cat', state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers='cat'),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=3, modify='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, int_points='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, raw_data='cat') ]
        assert( not any( tests ) )


    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
        nodes = self.mili.getNodes()

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
        state_maps = self.mili.getStateMaps()
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
        labels = self.mili.getLabels()
        
        PROC_0_LABELS = {'node': {66: 1, 67: 2, 68: 3, 70: 4, 71: 5, 72: 6, 74: 7, 75: 8, 76: 9, 78: 10, 79: 11, 80: 12, 82: 13, 83: 14, 84: 15, 86: 16, 87: 17, 88: 18, 90: 19, 91: 20, 92: 21, 94: 22, 95: 23, 96: 24, 98: 25, 99: 26, 100: 27, 102: 28, 103: 29, 104: 30, 106: 31, 107: 32, 108: 33, 111: 34, 112: 35}, 'brick': {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 13: 6, 14: 7, 15: 8, 16: 9, 17: 10, 18: 11}, 'glob': {1: 1},'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_1_LABELS = {'node': {90: 1, 91: 2, 94: 3, 95: 4, 98: 5, 99: 6, 100: 7, 102: 8, 103: 9, 104: 10, 106: 11, 107: 12, 108: 13, 110: 14, 111: 15, 112: 16, 114: 17, 115: 18, 116: 19, 118: 20, 119: 21, 120: 22, 122: 23, 123: 24, 124: 25, 126: 26, 127: 27, 128: 28, 130: 29, 131: 30, 132: 31, 134: 32, 135: 33, 136: 34, 138: 35, 139: 36, 140: 37, 142: 38, 143: 39, 144: 40}, 'brick': {12: 1, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6, 30: 7, 31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_2_LABELS = {'node': {4: 1, 6: 2, 13: 3, 14: 4, 15: 5, 58: 6, 59: 7, 60: 8, 63: 9, 64: 10, 101: 11, 102: 12, 105: 13, 106: 14, 109: 15, 110: 16, 117: 17, 118: 18, 121: 19, 122: 20, 125: 21, 126: 22, 137: 23, 138: 24, 141: 25, 142: 26}, 'beam': {5: 1, 6: 2}, 'brick': {21: 1, 23: 2, 24: 3}, 'shell': {9: 1, 11: 2, 12: 3}, 'cseg': {8: 1, 9: 2, 12: 3, 20: 4, 21: 5, 24: 6}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_3_LABELS = {'node': {12: 1, 13: 2, 14: 3, 15: 4, 49: 5, 50: 6, 51: 7, 52: 8, 53: 9, 54: 10, 55: 11, 56: 12, 65: 13, 66: 14, 69: 15, 70: 16, 73: 17, 74: 18, 77: 19, 78: 20, 81: 21, 82: 22, 85: 23, 86: 24, 89: 25, 90: 26, 93: 27, 94: 28, 97: 29, 98: 30, 101: 31, 102: 32, 105: 33, 106: 34, 109: 35, 110: 36}, 'brick': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, 'shell': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, 'cseg': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 12}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}} 
        PROC_4_LABELS = {'node': {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 12: 6, 13: 7, 16: 8, 21: 9, 22: 10, 25: 11, 29: 12, 30: 13, 31: 14, 32: 15, 33: 16, 34: 17, 37: 18, 38: 19, 39: 20}, 'beam': {2: 1, 3: 2, 12: 3, 13: 4, 17: 5, 22: 6, 23: 7, 24: 8, 25: 9, 27: 10, 28: 11, 32: 12, 33: 13, 34: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_5_LABELS = {'node': {1: 1, 2: 2, 3: 3, 12: 4, 13: 5, 57: 6, 58: 7, 59: 8, 61: 9, 62: 10, 63: 11, 97: 12, 98: 13, 101: 14, 102: 15, 113: 16, 114: 17, 117: 18, 118: 19, 121: 20, 122: 21, 129: 22, 130: 23, 133: 24, 134: 25, 137: 26, 138: 27}, 'beam': {1: 1, 4: 2}, 'brick': {19: 1, 20: 2, 22: 3}, 'shell': {7: 1, 8: 2, 10: 3}, 'cseg': {7: 1, 10: 2, 11: 3, 19: 4, 22: 5, 23: 6}, 'glob': {1: 1},'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_6_LABELS = {'node': {8: 1, 10: 2, 14: 3, 15: 4, 16: 5, 20: 6, 25: 7, 26: 8, 27: 9, 28: 10, 34: 11, 35: 12, 36: 13, 41: 14, 42: 15, 43: 16, 44: 17, 45: 18}, 'beam': {11: 1, 18: 2, 19: 3, 20: 4, 21: 5, 29: 6, 30: 7, 31: 8, 37: 9, 38: 10, 39: 11, 40: 12, 41: 13, 42: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_7_LABELS = {'node': {1: 1, 9: 2, 11: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 22: 9, 23: 10, 24: 11, 32: 12, 39: 13, 40: 14, 45: 15, 46: 16, 47: 17, 48: 18}, 'beam': {7: 1, 8: 2, 9: 3, 10: 4, 14: 5, 15: 6, 16: 7, 26: 8, 35: 9, 36: 10, 43: 11, 44: 12, 45: 13, 46: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}

        self.assertEqual(labels[0], PROC_0_LABELS)
        self.assertEqual(labels[1], PROC_1_LABELS)
        self.assertEqual(labels[2], PROC_2_LABELS)
        self.assertEqual(labels[3], PROC_3_LABELS)
        self.assertEqual(labels[4], PROC_4_LABELS)
        self.assertEqual(labels[5], PROC_5_LABELS)
        self.assertEqual(labels[6], PROC_6_LABELS)
        self.assertEqual(labels[7], PROC_7_LABELS)


    """
    Testing the getMaterials() method of the Mili class.
    """
    def test_materials_getter(self):
        materials = self.mili.getMaterials()

        PROC_0_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}
        PROC_1_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}}
        PROC_2_MATERIALS = {1: {'beam': [1, 2]}, 2: {'brick': [1, 2, 3]}, 3: {'shell': [1, 2, 3]}, 4: {'cseg': [1, 2, 3]}, 5: {'cseg': [4, 5, 6]}}
        PROC_3_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6]}, 3: {'shell': [1, 2, 3, 4, 5, 6]}, 4: {'cseg': [1, 2, 3, 4, 5, 6]}, 5: {'cseg': [7, 8, 9, 10, 11, 12]}}
        PROC_4_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}
        PROC_5_MATERIALS = {1: {'beam': [1, 2]}, 2: {'brick': [1, 2, 3]}, 3: {'shell': [1, 2, 3]}, 4: {'cseg': [1, 2, 3]}, 5: {'cseg': [4, 5, 6]}}
        PROC_6_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}
        PROC_7_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}

        self.assertEqual(materials[0], PROC_0_MATERIALS)
        self.assertEqual(materials[1], PROC_1_MATERIALS)
        self.assertEqual(materials[2], PROC_2_MATERIALS)
        self.assertEqual(materials[3], PROC_3_MATERIALS)
        self.assertEqual(materials[4], PROC_4_MATERIALS)
        self.assertEqual(materials[5], PROC_5_MATERIALS)
        self.assertEqual(materials[6], PROC_6_MATERIALS)
        self.assertEqual(materials[7], PROC_7_MATERIALS)
    

    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        state_variables = self.mili.getStateVariables()

        SVAR_PROC_0_CNT = 159
        SVAR_PROC_1_CNT = 126
        SVAR_PROC_2_CNT = 134
        SVAR_PROC_3_CNT = 134
        SVAR_PROC_4_CNT = 126
        SVAR_PROC_5_CNT = 134
        SVAR_PROC_6_CNT = 126
        SVAR_PROC_7_CNT = 126

        self.assertEqual(len(state_variables[0]), SVAR_PROC_0_CNT)
        self.assertEqual(len(state_variables[1]), SVAR_PROC_1_CNT)
        self.assertEqual(len(state_variables[2]), SVAR_PROC_2_CNT)
        self.assertEqual(len(state_variables[3]), SVAR_PROC_3_CNT)
        self.assertEqual(len(state_variables[4]), SVAR_PROC_4_CNT)
        self.assertEqual(len(state_variables[5]), SVAR_PROC_5_CNT)
        self.assertEqual(len(state_variables[6]), SVAR_PROC_6_CNT)
        self.assertEqual(len(state_variables[7]), SVAR_PROC_7_CNT)


    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.labels_of_material('es_13', False)
       self.assertEqual(len(answer.items), 12)
       for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].mo_id, i + 1)

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        self.assertEqual(len(answer.items), 46)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [13, 21, 23, 15, 14, 22, 24, 16]
        self.assertEqual(len(answer.items), 8)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matcgx'], 'mat', None, [1,2], [3], raw_data=False)
        self.assertEqual(len(answer.state_answers), 1)
        arr = [0.6021666526794434, 0.6706029176712036]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                self.assertEqual(answer.state_answers[i].items[j].value, arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
        
        answer = self.mili.query(['ux'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(0.4330127537250519, answer.state_answers[0].items[0].value)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [70], [4], raw_data=False)
        arr = [0.4330127537250519, 0.2500000596046448, 2.436666965484619]
        self.assertEqual(answer.state_answers[0].items[0].value, arr)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
        
        answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
    
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        val = {3 : {'matcgx' : {1 : 5.5}}}
        val2 = {3 : {'matcgx' : {1 : 0.6021666526794434}}}
    
        # Before change
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.6021666526794434)
        
        # After change
        self.mili.modify_state_variable('matcgx', 'mat', val, 1, [3])
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 5.5)
        
        # Back to original
        self.mili.modify_state_variable('matcgx', 'mat', val2, 1, [3])
        answer = self.mili.query('matcgx', 'mat', None, [1], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.6021666526794434)
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        val = {3 : {'nodpos' : {70 : [5.0, 6.0, 9.0], 71: [5.0, 6.0, 9.0]}}}
        val2 = {3 : {'nodpos' : {70 : [0.4330127537250519, 0.2500000596046448, 2.446666955947876], 71: [0.4330127239227295, 0.2499999850988388, 2.7133333683013916]}}}
        
        # Before change
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [0.4330127537250519, 0.2500000596046448, 2.446666955947876])
        self.assertEqual(answer.state_answers[0].items[1].value, [0.4330127239227295, 0.2499999850988388, 2.7133333683013916])
        
        
        # After change
        self.mili.modify_state_variable('nodpos', 'node', val, [70, 71], [3])
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [5.0, 6.0, 9.0])
        self.assertEqual(answer.state_answers[0].items[1].value, [5.0, 6.0, 9.0])
        
        # Back to original
        self.mili.modify_state_variable('nodpos', 'node', val2, [70, 71], [3])
        answer = self.mili.query('nodpos', 'node', None, [70, 71], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, [0.4330127537250519, 0.2500000596046448, 2.446666955947876])
        self.assertEqual(answer.state_answers[0].items[1].value, [0.4330127239227295, 0.2499999850988388, 2.7133333683013916])
    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        val = {4 : {'nodpos[uz]' : {70 : 9.0, 71: 9.0}}}
        val2 = {4 : {'nodpos[uz]' : {70 : 2.436666965484619, 71 : 2.7033333778381348}}}
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 2.436666965484619)
        self.assertEqual(answer.state_answers[0].items[1].value, 2.7033333778381348)
    
        
        # After change
        self.mili.modify_state_variable('nodpos[uz]', 'node', val, [70, 71], [4])
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 9.0)
        self.assertEqual(answer.state_answers[0].items[1].value, 9.0)
    
        
        # Back to original
        self.mili.modify_state_variable('nodpos[uz]', 'node', val2, [70, 71], [4])
        answer = self.mili.query('nodpos[uz]', 'node', None, [70, 71], [4], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 2.436666965484619)
        self.assertEqual(answer.state_answers[0].items[1].value, 2.7033333778381348)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [21,22], False, [3], False)
        d= {'sx': {3: -899.3980102539062}, 'sy': {3: -893.2693481445312}, 'sz': {3: 9.335740287497174e-06}, 'sxy': {3: 896.3336791992188}, 'syz': {3: -2.961223840713501}, 'szx': {3: 2.9612176418304443}}
        
        self.assertEqual(answer.state_answers[1].items[0].value, d)
   
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        d = {'sy' : {2: -5373.53173828125}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)
    
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
        d = {70 : {'stress' : {5 : {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}}}}
        dd = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)
        
        mod = {70 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)

    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        d = {70 : {'stress' : {5 : {'sy': {2: -5373.53173828125}}}}}
        dd = -5373.53173828125
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)
        
        # After change
        mod = {70 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)


'''
Testing the parallel Mili file version
'''
class TestMiliParallelReaderThreaded(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
        file_name = 'parallel/d3samp6.plt'
        self.mili = reader.Mili()
        self.mili.read(file_name, parallel_read=True)
        self.mili.setErrorFile()   

    def tearDown(self):
        # Ensures that created processes are all killed at end of testing
        self.mili.closeAllConnections()
    
    '''
     Testing invalid inputs
     '''
    def test_invalid_inputs(self):
        tests = [ self.mili.labels_of_material('es_13121'),
                  self.mili.nodes_of_material('es_13121'),
                  self.mili.nodes_of_elem('1', 'bsseam'),
                  self.mili.nodes_of_elem('-1', 'beam'),
                   self.mili.query(['nodpos[uxxx]'], 'node', labels=4, state_numbers=3),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'nodels', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'node', material='cat', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'node', labels=-4,state_numbers= 3),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 4, labels=4, state_numbers=300),
                   self.mili.query([4], 'node', labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', 9, labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', labels='cat', state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers='cat'),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=3, modify='cat'),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, int_points='cat'),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, raw_data='cat') ]
        assert( not any( tests ) )


    """
    Testing the getNodes() method of the Mili class.
    """
    def test_nodes_getter(self):
        nodes = self.mili.getNodes()

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
        state_maps = self.mili.getStateMaps()
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
        labels = self.mili.getLabels()
        
        PROC_0_LABELS = {'node': {66: 1, 67: 2, 68: 3, 70: 4, 71: 5, 72: 6, 74: 7, 75: 8, 76: 9, 78: 10, 79: 11, 80: 12, 82: 13, 83: 14, 84: 15, 86: 16, 87: 17, 88: 18, 90: 19, 91: 20, 92: 21, 94: 22, 95: 23, 96: 24, 98: 25, 99: 26, 100: 27, 102: 28, 103: 29, 104: 30, 106: 31, 107: 32, 108: 33, 111: 34, 112: 35},'brick': {7: 1, 8: 2, 9: 3, 10: 4, 11: 5, 13: 6, 14: 7, 15: 8, 16: 9, 17: 10, 18: 11}, 'glob': {1: 1},'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_1_LABELS = {'node': {90: 1, 91: 2, 94: 3, 95: 4, 98: 5, 99: 6, 100: 7, 102: 8, 103: 9, 104: 10, 106: 11, 107: 12, 108: 13, 110: 14, 111: 15, 112: 16, 114: 17, 115: 18, 116: 19, 118: 20, 119: 21, 120: 22, 122: 23, 123: 24, 124: 25, 126: 26, 127: 27, 128: 28, 130: 29, 131: 30, 132: 31, 134: 32, 135: 33, 136: 34, 138: 35, 139: 36, 140: 37, 142: 38, 143: 39, 144: 40},'brick': {12: 1, 25: 2, 26: 3, 27: 4, 28: 5, 29: 6, 30: 7, 31: 8, 32: 9, 33: 10, 34: 11, 35: 12, 36: 13},'glob': {1: 1},'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_2_LABELS = {'node': {4: 1, 6: 2, 13: 3, 14: 4, 15: 5, 58: 6, 59: 7, 60: 8, 63: 9, 64: 10, 101: 11, 102: 12, 105: 13, 106: 14, 109: 15, 110: 16, 117: 17, 118: 18, 121: 19, 122: 20, 125: 21, 126: 22, 137: 23, 138: 24, 141: 25, 142: 26}, 'beam': {5: 1, 6: 2}, 'brick': {21: 1, 23: 2, 24: 3}, 'shell': {9: 1, 11: 2, 12: 3}, 'cseg': {8: 1, 9: 2, 12: 3, 20: 4, 21: 5, 24: 6}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_3_LABELS = {'node': {12: 1, 13: 2, 14: 3, 15: 4, 49: 5, 50: 6, 51: 7, 52: 8, 53: 9, 54: 10, 55: 11, 56: 12, 65: 13, 66: 14, 69: 15, 70: 16, 73: 17, 74: 18, 77: 19, 78: 20, 81: 21, 82: 22, 85: 23, 86: 24, 89: 25, 90: 26, 93: 27, 94: 28, 97: 29, 98: 30, 101: 31, 102: 32, 105: 33, 106: 34, 109: 35, 110: 36}, 'brick': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},'shell': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6},'cseg': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 12}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}} 
        PROC_4_LABELS = {'node': {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 12: 6, 13: 7, 16: 8, 21: 9, 22: 10, 25: 11, 29: 12, 30: 13, 31: 14, 32: 15, 33: 16, 34: 17, 37: 18, 38: 19, 39: 20}, 'beam': {2: 1, 3: 2, 12: 3, 13: 4, 17: 5, 22: 6, 23: 7, 24: 8, 25: 9, 27: 10, 28: 11, 32: 12, 33: 13, 34: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_5_LABELS = {'node': {1: 1, 2: 2, 3: 3, 12: 4, 13: 5, 57: 6, 58: 7, 59: 8, 61: 9, 62: 10, 63: 11, 97: 12, 98: 13, 101: 14, 102: 15, 113: 16, 114: 17, 117: 18, 118: 19, 121: 20, 122: 21, 129: 22, 130: 23, 133: 24, 134: 25, 137: 26, 138: 27}, 'beam': {1: 1, 4: 2}, 'brick': {19: 1, 20: 2, 22: 3}, 'shell': {7: 1, 8: 2, 10: 3},'cseg': {7: 1, 10: 2, 11: 3, 19: 4, 22: 5, 23: 6},'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_6_LABELS = {'node': {8: 1, 10: 2, 14: 3, 15: 4, 16: 5, 20: 6, 25: 7, 26: 8, 27: 9, 28: 10, 34: 11, 35: 12, 36: 13, 41: 14, 42: 15, 43: 16, 44: 17, 45: 18}, 'beam': {11: 1, 18: 2, 19: 3, 20: 4, 21: 5, 29: 6, 30: 7, 31: 8, 37: 9, 38: 10, 39: 11, 40: 12, 41: 13, 42: 14}, 'glob': {1: 1},'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}
        PROC_7_LABELS = {'node': {1: 1, 9: 2, 11: 3, 16: 4, 17: 5, 18: 6, 19: 7, 20: 8, 22: 9, 23: 10, 24: 11, 32: 12, 39: 13, 40: 14, 45: 15, 46: 16, 47: 17, 48: 18}, 'beam': {7: 1, 8: 2, 9: 3, 10: 4, 14: 5, 15: 6, 16: 7, 26: 8, 35: 9, 36: 10, 43: 11, 44: 12, 45: 13, 46: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}

        self.assertEqual(labels[0], PROC_0_LABELS)
        self.assertEqual(labels[1], PROC_1_LABELS)
        self.assertEqual(labels[2], PROC_2_LABELS)
        self.assertEqual(labels[3], PROC_3_LABELS)
        self.assertEqual(labels[4], PROC_4_LABELS)
        self.assertEqual(labels[5], PROC_5_LABELS)
        self.assertEqual(labels[6], PROC_6_LABELS)
        self.assertEqual(labels[7], PROC_7_LABELS)


    """
    Testing the getMaterials() method of the Mili class.
    """
    def test_materials_getter(self):
        materials = self.mili.getMaterials()

        PROC_0_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]}}
        PROC_1_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]}}
        PROC_2_MATERIALS = {1: {'beam': [1, 2]}, 2: {'brick': [1, 2, 3]}, 3: {'shell': [1, 2, 3]}, 4: {'cseg': [1, 2, 3]}, 5: {'cseg': [4, 5, 6]}}
        PROC_3_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6]}, 3: {'shell': [1, 2, 3, 4, 5, 6]}, 4: {'cseg': [1, 2, 3, 4, 5, 6]}, 5: {'cseg': [7, 8, 9, 10, 11, 12]}}
        PROC_4_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}
        PROC_5_MATERIALS = {1: {'beam': [1, 2]}, 2: {'brick': [1, 2, 3]}, 3: {'shell': [1, 2, 3]}, 4: {'cseg': [1, 2, 3]}, 5: {'cseg': [4, 5, 6]}}
        PROC_6_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}
        PROC_7_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}

        self.assertEqual(materials[0], PROC_0_MATERIALS)
        self.assertEqual(materials[1], PROC_1_MATERIALS)
        self.assertEqual(materials[2], PROC_2_MATERIALS)
        self.assertEqual(materials[3], PROC_3_MATERIALS)
        self.assertEqual(materials[4], PROC_4_MATERIALS)
        self.assertEqual(materials[5], PROC_5_MATERIALS)
        self.assertEqual(materials[6], PROC_6_MATERIALS)
        self.assertEqual(materials[7], PROC_7_MATERIALS)
    

    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        state_variables = self.mili.getStateVariables()

        SVAR_PROC_0_CNT = 159
        SVAR_PROC_1_CNT = 126
        SVAR_PROC_2_CNT = 134
        SVAR_PROC_3_CNT = 134
        SVAR_PROC_4_CNT = 126
        SVAR_PROC_5_CNT = 134
        SVAR_PROC_6_CNT = 126
        SVAR_PROC_7_CNT = 126

        self.assertEqual(len(state_variables[0]), SVAR_PROC_0_CNT)
        self.assertEqual(len(state_variables[1]), SVAR_PROC_1_CNT)
        self.assertEqual(len(state_variables[2]), SVAR_PROC_2_CNT)
        self.assertEqual(len(state_variables[3]), SVAR_PROC_3_CNT)
        self.assertEqual(len(state_variables[4]), SVAR_PROC_4_CNT)
        self.assertEqual(len(state_variables[5]), SVAR_PROC_5_CNT)
        self.assertEqual(len(state_variables[6]), SVAR_PROC_6_CNT)
        self.assertEqual(len(state_variables[7]), SVAR_PROC_7_CNT)


    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
        answer = self.mili.labels_of_material('es_13', False)
        self.assertEqual(len(answer.items), 12)
        for i in range(len(answer.items)):
            self.assertEqual(answer.items[i].mo_id, i + 1)

    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        self.assertEqual(len(answer.items), 46)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [13, 21, 23, 15, 14, 22, 24, 16]
        self.assertEqual(len(answer.items), 8)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matcgx'], 'mat', None, [1,2], [4], raw_data=False)
        self.assertEqual(len(answer.state_answers), 1)
        arr = [0.6021666526794434, 0.6706029176712036]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                self.assertEqual(answer.state_answers[i].items[j].value, arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
        
        answer = self.mili.query(['ux'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(0.4330127537250519, answer.state_answers[0].items[0].value)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [70], [3], raw_data=False)
        arr = [0.4330127537250519, 0.2500000596046448, 2.446666955947876]
        self.assertEqual(answer.state_answers[0].items[0].value, arr)

    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
        
        answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        d = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        d = {'sy': {2: -5373.53173828125}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)

    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
        d = {70 : {'stress' : {5 : {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}}}}
        dd = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)
        
        mod = {70 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)

    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        d = {70 : {'stress' : {5 : {'sy': {2: -5373.53173828125}}}}}
        dd = -5373.53173828125
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)
        
        # After change
        mod = {70 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)


'''
Testing the Mili Reader in Parallel with more "A" files than processors
'''
class TestMiliParallelReaderThreadedSetProcessorCount(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
        file_name = 'parallel/d3samp6.plt'
        self.mili = reader.Mili()
        self.mili.read(file_name, parallel_read=True, processors=2)
        self.mili.setErrorFile()   

    def tearDown(self):
        # Ensures that created processes are all killed at end of testing
        self.mili.closeAllConnections()
    
    '''
     Testing invalid inputs
     '''
    def test_invalid_inputs(self):
        tests = [ self.mili.labels_of_material('es_13121'),
                  self.mili.nodes_of_material('es_13121'),
                  self.mili.nodes_of_elem('1', 'bsseam'),
                  self.mili.nodes_of_elem('-1', 'beam'),
                   self.mili.query(['nodpos[uxxx]'], 'node', labels=4, state_numbers=3),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'nodels', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'node', material='cat', labels=4, state_numbers=3),
                   self.mili.query(['nodpos[ux]'], 'node', labels=-4,state_numbers= 3),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 4, labels=4, state_numbers=300),
                   self.mili.query([4], 'node', labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', 9, labels=4, state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', labels='cat', state_numbers=300),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers='cat'),
                   self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=3, modify='cat'),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, int_points='cat'),
                   self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, raw_data='cat') ]
        assert( not any( tests ) )
    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
        answer = self.mili.labels_of_material('es_13', False)
        self.assertEqual(len(answer.items), 12)
        for i in range(len(answer.items)):
            self.assertEqual(answer.items[i].mo_id, i + 1)

    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        self.assertEqual(len(answer.items), 46)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [13, 21, 23, 15, 14, 22, 24, 16]
        self.assertEqual(len(answer.items), 8)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matcgx'], 'mat', None, [1,2], [4], raw_data=False)
        self.assertEqual(len(answer.state_answers), 1)
        arr = [0.6021666526794434, 0.6706029176712036]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                self.assertEqual(answer.state_answers[i].items[j].value, arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
        
        answer = self.mili.query(['ux'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(0.4330127537250519, answer.state_answers[0].items[0].value)

    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [70], [3], raw_data=False)
        arr = [0.4330127537250519, 0.2500000596046448, 2.446666955947876]
        self.assertEqual(answer.state_answers[0].items[0].value, arr)

    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)
        
        answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 36)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        d = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        d = {'sy': {2: -5373.53173828125}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)

    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
        d = {70 : {'stress' : {5 : {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}}}}
        dd = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)
        
        mod = {70 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value, dd)

    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        d = {70 : {'stress' : {5 : {'sy': {2: -5373.53173828125}}}}}
        dd = -5373.53173828125
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)
        
        # After change
        mod = {70 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        self.assertEqual(answer.state_answers[0].items[0].value['sy'][2], dd)


'''
Testing the Mili Reader when reading in a subset of "A" files.
'''
class TestMiliReaderWithSubsetOfAFiles(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
        file_name = 'parallel/d3samp6.plt'
        self.mili = reader.Mili()
        self.mili.read(file_name, parallel_read=True, a_files=[3,4])
        self.mili.setErrorFile()   

    def tearDown(self):
        # Ensures that created processes are all killed at end of testing
        self.mili.closeAllConnections()

    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        tests = [ self.mili.labels_of_material('es_13121'),
                  self.mili.nodes_of_material('es_13121'),
                  self.mili.nodes_of_elem('1', 'bsseam'),
                  self.mili.nodes_of_elem('-1', 'beam'),
                  self.mili.query(['nodpos[uxxx]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'nodels', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', material='cat', labels=4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=-4, state_numbers=3),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 4, labels=4, state_numbers=300),
                  self.mili.query([4], 'node', labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', material=9, labels=4, state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels='cat', state_numbers=300),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers='cat'),
                  self.mili.query(['nodpos[ux]'], 'node', labels=4, state_numbers=3, modify='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, int_points='cat'),
                  self.mili.query(['nodposss[ux]'], 'node', labels=4, state_numbers=3, modify=False, raw_data='cat') ]
        assert( not any( tests ) )

    """
    Testing Labels getter
    """
    def test_get_labels(self):
        PROC_3_LABELS = {'node': {12: 1, 13: 2, 14: 3, 15: 4, 49: 5, 50: 6, 51: 7, 52: 8, 53: 9, 54: 10, 55: 11, 56: 12, 65: 13, 66: 14, 69: 15, 70: 16, 73: 17, 74: 18, 77: 19, 78: 20, 81: 21, 82: 22, 85: 23, 86: 24, 89: 25, 90: 26, 93: 27, 94: 28, 97: 29, 98: 30, 101: 31, 102: 32, 105: 33, 106: 34, 109: 35, 110: 36}, 'brick': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, 'shell': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}, 'cseg': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 13: 7, 14: 8, 15: 9, 16: 10, 17: 11, 18: 12}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}} 
        PROC_4_LABELS = {'node': {3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 12: 6, 13: 7, 16: 8, 21: 9, 22: 10, 25: 11, 29: 12, 30: 13, 31: 14, 32: 15, 33: 16, 34: 17, 37: 18, 38: 19, 39: 20}, 'beam': {2: 1, 3: 2, 12: 3, 13: 4, 17: 5, 22: 6, 23: 7, 24: 8, 25: 9, 27: 10, 28: 11, 32: 12, 33: 13, 34: 14}, 'glob': {1: 1}, 'mat': {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}}

        labels = self.mili.getLabels()
        
        self.assertEqual(labels[0], PROC_3_LABELS)
        self.assertEqual(labels[1], PROC_4_LABELS)

    """
    Testing the getMaterials() method of the Mili class.
    """
    def test_materials_getter(self):
        materials = self.mili.getMaterials()

        PROC_3_MATERIALS = {2: {'brick': [1, 2, 3, 4, 5, 6]}, 3: {'shell': [1, 2, 3, 4, 5, 6]}, 4: {'cseg': [1, 2, 3, 4, 5, 6]}, 5: {'cseg': [7, 8, 9, 10, 11, 12]}}
        PROC_4_MATERIALS = {1: {'beam': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]}}

        self.assertEqual(materials[0], PROC_3_MATERIALS)
        self.assertEqual(materials[1], PROC_4_MATERIALS)
    

    """
    Testing the getStateVariables() method of the Mili class.
    """
    def test_statevariables_getter(self):
        state_variables = self.mili.getStateVariables()

        SVAR_PROC_3_CNT = 134
        SVAR_PROC_4_CNT = 126

        self.assertEqual(len(state_variables[0]), SVAR_PROC_3_CNT)
        self.assertEqual(len(state_variables[1]), SVAR_PROC_4_CNT)


    """
    Test querying for results that do not exist on processor files 3 or 4
    """
    def test_query_invalid_results(self):
        invalid_queries = [
            self.mili.query("ax", "node", labels=[111,112,113]),
            self.mili.query("sx", "brick", labels=[7,8,9]),
            self.mili.query("sx", "shell", labels=[7,8,9]),
            self.mili.query("axf", "beam", labels=[1,4,5,6,]),
            self.mili.nodes_of_elem(7, "brick", False)
        ]
        assert( not any(invalid_queries) )

    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False, global_ids=True)
        arr = [65, 81, 85, 69, 66, 82, 86, 70]
        self.assertEqual(len(answer.items), 8)
        for i in range(len(answer.items)):
           self.assertEqual(answer.items[i].label, arr[i])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
        
        answer = self.mili.query(['ux'], 'node', None, [70], [3], None, None, False)
        self.assertEqual(answer.state_answers[0].items[0].value, 0.4330127537250519)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 6)
        
        answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
        self.assertEqual(len(answer.state_answers[0].items), 6)

    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels=[12], state_numbers=[70], int_points=[2], raw_data=False)
        d = {'sx': {2: 395.00811767578125}, 'sy': {2: 118.7711410522461}, 'sz': {2: 6088.36376953125}, 'sxy': {2: 216.87445068359375}, 'syz': {2: 1735.9781494140625}, 'szx': {2: 3308.8525390625}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)

    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [12], [70], False, [2], False)
        d = {'sy': {2: 118.7711410522461}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)

# Run the testing
if __name__ == '__main__':
    unittest.main()
