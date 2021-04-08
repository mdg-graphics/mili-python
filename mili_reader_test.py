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
        # add more here
        
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
    # ANSWER UPDATED
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [21,22], False, [2], False)
        # OLD ANSWER: base_answer = {'sx': {2: 303.8838806152344}, 'sy': {2: -0.33932867646217346}, 'sz': {2: -6.556616085617861e-07}, 'sxy': {2: 0.0}, 'syz': {2: 896.3336791992188}, 'szx': {2: 426.0271301269531}}
        base_answer = {'sx': {2: -1018.4232177734375}, 'sy': {2: -1012.2537231445312}, 'sz': {2: -6.556616085617861e-07}, 'sxy': {2: 1015.3384399414062}, 'syz': {2: 0.3263571858406067}, 'szx': {2: -0.32636013627052307}}
        self.assertEqual(answer.state_answers[1].items[0].value, base_answer)
    
    '''
    Test accessing a vector array component
    '''
    # ANSWER UPDATED
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [71], False, [2], False)
        # OLD ANSWER: base_answer = {'sy' : {2: -0.040591922402381897}}
        base_answer = {'sy' : {2: -5545.70751953125}}
        self.assertEqual(answer.state_answers[0].items[0].value, base_answer)
        
    '''
    Test modifying a vector array 
    '''
    # ANSWER UPDATED
    def test_modify_vector_array(self):
         # Before change
        # OLD ORIGINAL VALUES: d = {71 : {'stress' : {5 : {'sz': {2: -3.736035978363361e-07}, 'sy': {2: -0.40591922402381897}, 'sx': {2: 6544.61962890625}, 'szx': {2: 6944.44775390625}, 'sxy': {2: 0.0}, 'syz': {2: 5146.49267578125}}}}}
        # OLD ANSWER: dd = {'sz': {2: -3.736035978363361e-07}, 'sy': {2: -0.40591922402381897}, 'sx': {2: 6544.61962890625}, 'szx': {2: 6944.44775390625}, 'sxy': {2: 0.0}, 'syz': {2: 5146.49267578125}}

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
    # ANSWER UPDATED
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [21,22], False, [3], False)
        
        # OLD ANSWER: d= {'sx': {3: -6.81397864354949e-07}, 'sy': {3: 0.0}, 'sz': {3: 1015.3384399414062}, 'sxy': {3: -899.3980102539062}, 'syz': {3: -2.961223840713501}, 'szx': {3: 422.8572082519531}}
        d= {'sx': {3: -899.3980102539062}, 'sy': {3: -893.2693481445312}, 'sz': {3: 9.335740287497174e-06}, 'sxy': {3: 896.3336791992188}, 'syz': {3: -2.961223840713501}, 'szx': {3: 2.9612176418304443}}
        
        self.assertEqual(answer.state_answers[1].items[0].value, d)
   
    '''
    Test accessing a vector array component
    '''
    # ANSWER UPDATED
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        # OLD ANSWER: d = {'sy' : {2: -0.6875680685043335}}
        # Answer in .hsp file is -5373.479298375017
        d = {'sy' : {2: -5373.53173828125}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)
    
    '''
    Test modifying a vector array 
    '''
    # ANSWER UPDATED
    def test_modify_vector_array(self):
    # OLD ORIGINAL VALUE   d = {70 : {'stress' : {5 : {'sz': {2: -2.4870882953109685e-07}, 'sy': {2: -0.2791216969490051}, 'sx': {2: 6544.6015625}, 'szx': {2: 6944.56103515625}, 'sxy': {2: 0.0}, 'syz': {2: 5146.34326171875}}}}}
    # OLD ANSWER   dd = {'sz': {2: -2.4870882953109685e-07}, 'sy': {2: -0.2791216969490051}, 'sx': {2: 6544.6015625}, 'szx': {2: 6944.56103515625}, 'sxy': {2: 0.0}, 'syz': {2: 5146.34326171875}}

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
        # OLD ORIGINAL VALUE d = {70 : {'stress' : {5 : {'sy': {2: -0.2791216969490051}}}}}
        # OLD ANSWER dd = -0.2791216969490051
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
        self.mili.read(file_name, parallel_read=False)
        self.mili.setErrorFile()   

    def tearDown(self):
        # Ensures that created processes are all killed at end of testing
        del self.mili
    
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
    # ANSWER UPDATED 
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        #d = {'sz': {2: -2.4870882953109685e-07}, 'sy': {2: -0.2791216969490051}, 'sx': {2: 6544.6015625}, 'szx': {2: 6944.56103515625}, 'sxy': {2: 0.0}, 'syz': {2: 5146.34326171875}}
        # OLD ANSWER: d = {'sx': {2: 5081.48583984375}, 'sy': {2: -0.6875680685043335}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 0.0}, 'syz': {2: 5818.755859375}, 'szx': {2: 4640.2470703125}}
        d = {'sx': {2: -5377.6376953125}, 'sy': {2: -5373.53173828125}, 'sz': {2: -3.930831553589087e-07}, 'sxy': {2: 5375.58447265625}, 'syz': {2: 0.6931889057159424}, 'szx': {2: -0.693189263343811}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)
    
    '''
    Test accessing a vector array component
    '''
    # ANSWER UPDATED
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        d = {'sy': {2: -5373.53173828125}}
        self.assertEqual(answer.state_answers[0].items[0].value, d)

if __name__ == '__main__':
    unittest.main()
