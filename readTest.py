import unittest
import mili_reader_lib as read
import os, sys
import psutil
'''
These are tests to assert the correctness of the Mili Reader

These tests use d3samp6.dyn
'''
class TestMiliReader(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):        
    	file_name = 'd3samp6.plt'
        #file_name = 'states/d3samp6.plt'    
        self.mili = read.Mili(file_name)
        self.mili.setErrorFile()
	
    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        answer = self.mili.labels_of_material('es_13121')
        answer = self.mili.nodes_of_material('es_13121', 'beam')
        answer = self.mili.nodes_of_material('es_1', 'bsseam')
        answer = self.mili.nodes_of_elem('1', 'bsseam')
        answer = self.mili.nodes_of_elem('-1', 'beam')
        answer = self.mili.query(['nodpos[uxxx]'], 'node', None, 4, 3)
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'nodels', None, 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', 'cat', 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, -4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 4, None, 4, 300)
        answer = self.mili.query([4], 'node', None, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', 9, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 'cat', 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 'cat')
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 3, 'cat')
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3, False, 'cat')
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3, False, None, 'cat')

        
        # add more here
        
    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.labels_of_material('es_13', False)
       assert(len(answer.items) == 12)
       for i in range(len(answer.items)):
           assert(answer.items[i].mo_id == i + 1)

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        assert(len(answer.items) == 48)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [65, 81, 85, 69, 66, 82, 86, 70]
        assert(len(answer.items) == 8)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matke'], 'mat', None, [1,2], [3, 4], raw_data=False)
        assert(len(answer.state_answers) == 2)
        arr = [0.0, 1662.0]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                assert(answer.state_answers[i].items[j].value == arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [4], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.9659258127212524)
        
        answer = self.mili.query(['ux'], 'node', None, [4], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.9659258127212524)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
	assert(len(answer.state_answers[0].items) == 36)
	
	answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
	assert(len(answer.state_answers[0].items) == 36)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3], None, None, False)
        arr = [0.9659258127212524, 0.4330127537250519, 1.000000238418579]
        assert(answer.state_answers[0].items[0].value == arr)
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        val = {3 : {'matke' : {1 : 5.5}}}
        val2 = {3 : {'matke' : {1 : 0.0}}}
    
        # Before change
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
        
        # After change
        self.mili.modify_state_variable('matke', 'mat', val, 1, [3])
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 5.5)
        
        # Back to original
        self.mili.modify_state_variable('matke', 'mat', val2, 1, [3])
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        val = {3 : {'nodpos' : {4 : [5.0, 6.0, 9.0], 5: [5.0, 6.0, 9.0]}}}
        val2 = {3 : {'nodpos' : {4 : [0.9659258127212524, 0.4330127537250519, 1.000000238418579], 5 : [0.258819043636322, 0.25000011920928955, 9.362678099478217e-08]}}}
        
        # Before change
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == [0.9659258127212524, 0.4330127537250519, 1.000000238418579])
        assert(answer.state_answers[0].items[1].value == [0.258819043636322, 0.25000011920928955, 9.362678099478217e-08])
        
        
        # After change
        self.mili.modify_state_variable('nodpos', 'node', val, [4, 5], [3])
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value ==[5.0, 6.0, 9.0])
        assert(answer.state_answers[0].items[1].value ==[5.0, 6.0, 9.0])
        
        # Back to original
        self.mili.modify_state_variable('nodpos', 'node', val2, [4, 5], [3])
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value ==[0.9659258127212524, 0.4330127537250519, 1.000000238418579])
        assert(answer.state_answers[0].items[1].value ==[0.258819043636322, 0.25000011920928955, 9.362678099478217e-08])
    
    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        val = {3 : {'nodpos[uz]' : {4 : 9.0, 5: 9.0}}}
        val2 = {3 : {'nodpos[uz]' : {4 : 1.000000238418579, 5 : 9.362678099478217e-08}}}
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 1.000000238418579)
        assert(answer.state_answers[0].items[1].value == 9.362678099478217e-08)
    
        
        # After change
        self.mili.modify_state_variable('nodpos[uz]', 'node', val, [4, 5], [3])
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 9.0)
        assert(answer.state_answers[0].items[1].value == 9.0)
    
        
        # Back to original
        self.mili.modify_state_variable('nodpos[uz]', 'node', val2, [4, 5], [3])
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 1.000000238418579)
        assert(answer.state_answers[0].items[1].value == 9.362678099478217e-08)
        
    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        d = {'sz': {2: -5145.794921875}, 'sy': {2: -5545.70751953125}, 'sx': {2: 6544.61962890625}, 'szx': {2: -748.0955810546875}, 'sxy': {2: 6944.18017578125}, 'syz': {2: 1537.8297119140625}}
        assert(answer.state_answers[0].items[0].value == d)
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == {'sy' : {2: -5545.70751953125}})
        
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
         # Before change
        d = {70 : {'stress' : {5 : {'sz': {2: -5145.794921875}, 'sy': {2: -5545.70751953125}, 'sx': {2: 6544.61962890625}, 'szx': {2: -748.0955810546875}, 'sxy': {2: 6944.18017578125}, 'syz': {2: 1537.8297119140625}}}}}
        dd = {'sz': {2: -5145.794921875}, 'sy': {2: -5545.70751953125}, 'sx': {2: 6544.61962890625}, 'szx': {2: -748.0955810546875}, 'sxy': {2: 6944.18017578125}, 'syz': {2: 1537.8297119140625}}
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == dd)
        
        mod = {70 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == dd)
    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        d = {70 : {'stress' : {5 : {'sy': {2: -5545.70751953125}}}}}
        dd = -5545.70751953125
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == dd)
        
        # After change
        mod = {70 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == dd)

'''
Testing the parallel Mili file version
'''
class TestMiliReaderParallel(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
        file_name = 'parallel/d3samp6.plt'
        self.mili = read.Mili()
	self.mili.read(file_name, parallel_read=True)
	self.mili.setErrorFile()   
    
    '''
    Testing invalid inputs
    '''
    def test_invalid_inputs(self):
        answer = self.mili.labels_of_material('es_13121')
        answer = self.mili.nodes_of_material('es_13121', 'beam')
        answer = self.mili.nodes_of_material('es_1', 'bsseam')
        answer = self.mili.nodes_of_elem('1', 'bsseam')
        answer = self.mili.nodes_of_elem('-1', 'beam')
        answer = self.mili.query(['nodpos[uxxx]'], 'node', None, 4, 3)
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'nodels', None, 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', 'cat', 4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, -4, 3)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 4, None, 4, 300)
        answer = self.mili.query([4], 'node', None, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', 9, 4, 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 'cat', 300)
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 'cat')
        answer = self.mili.query(['nodpos[ux]'], 'node', None, 4, 3, 'cat')
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3, False, 'cat')
        answer = self.mili.query(['nodposss[ux]'], 'node', None, 4, 3, False, None, 'cat')
        
    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_labels_material(self):
       answer = self.mili.labels_of_material('es_13', False)
       assert(len(answer.items) == 12)
       for i in range(len(answer.items)):
           assert(answer.items[i].mo_id == i + 1)

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1', False)
        assert(len(answer.items) == 48)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, 'brick', False)
        arr = [13, 21, 23, 15, 14, 22, 24, 16]
        assert(len(answer.items) == 8)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_variable(self):
        answer = self.mili.query(['matke'], 'mat', None, [1,2], [3, 4], raw_data=False)
        assert(len(answer.state_answers) == 2)
        arr = [0.0, 1662.0]
        for i in range(len(answer.state_answers)):
            for j in range(len(answer.state_answers[i].items)):
                assert(answer.state_answers[i].items[j].value == arr[j])
    
    '''
    Testing accessing accessing node attributes -> this is a vector component
    Tests both ways of accessing vector components (using brackets vs not)
    *Note* this is another case of state variable
    '''
    def test_node_attributes(self):
        answer = self.mili.query(['nodpos[ux]'], 'node', None, [4], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == -3.2783542991410286e-08)
        
        answer = self.mili.query(['ux'], 'node', None, [4], [3], None, None, False)
        assert(-3.2783542991410286e-08 == answer.state_answers[0].items[0].value)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3], raw_data=False)
        arr = [-3.2783542991410286e-08, 2.9700000286102295, 0.0]
        assert(answer.state_answers[0].items[0].value == arr)
    
    '''
    Test querying by material name and number:
    '''
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', 2, None, [37], raw_data=False)
	assert(len(answer.state_answers[0].items) == 36)
	
	answer = self.mili.query('sx', 'brick', 'es_12', None, [37], raw_data=False)
	assert(len(answer.state_answers[0].items) == 36)
    
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        val = {3 : {'matke' : {1 : 5.5}}}
        val2 = {3 : {'matke' : {1 : 0.0}}}
    
        # Before change
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
        
        # After change
        self.mili.modify_state_variable('matke', 'mat', val, 1, [3])
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 5.5)
        
        # Back to original
        self.mili.modify_state_variable('matke', 'mat', val2, 1, [3])
        answer = self.mili.query('matke', 'mat', None, [1], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        val = {3 : {'nodpos' : {4 : [5.0, 6.0, 9.0], 5: [5.0, 6.0, 9.0]}}}
        val2 = {3 : {'nodpos' : {4 : [-3.2783542991410286e-08, 2.9700000286102295, 0.0], 5 : [0.75, 0.0, -1000.0]}}}
        
        # Before change
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == [-3.2783542991410286e-08, 2.9700000286102295, 0.0])
        assert(answer.state_answers[0].items[1].value == [0.75, 0.0, -1000.0])
        
        
        # After change
        self.mili.modify_state_variable('nodpos', 'node', val, [4, 5], [3])
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value ==[5.0, 6.0, 9.0])
        assert(answer.state_answers[0].items[1].value ==[5.0, 6.0, 9.0])
        
        # Back to original
        self.mili.modify_state_variable('nodpos', 'node', val2, [4, 5], [3])
        answer = self.mili.query('nodpos', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value ==[-3.2783542991410286e-08, 2.9700000286102295, 0.0])
        assert(answer.state_answers[0].items[1].value ==[0.75, 0.0, -1000.0])
    
    '''
    Testing the modification of a vector component
    '''
    def test_modify_vector_component(self):
        val = {3 : {'nodpos[uz]' : {4 : 9.0, 5: 9.0}}}
        val2 = {3 : {'nodpos[uz]' : {4 : 0.0, 5 : -1000.0}}}
        
        # Before change
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
        assert(answer.state_answers[0].items[1].value == -1000.0)
    
        
        # After change
        self.mili.modify_state_variable('nodpos[uz]', 'node', val, [4, 5], [3])
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 9.0)
        assert(answer.state_answers[0].items[1].value == 9.0)
    
        
        # Back to original
        self.mili.modify_state_variable('nodpos[uz]', 'node', val2, [4, 5], [3])
        answer = self.mili.query('nodpos[uz]', 'node', None, [4, 5], [3], None, None, False)
        assert(answer.state_answers[0].items[0].value == 0.0)
        assert(answer.state_answers[0].items[1].value == -1000.0)
			
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):

        d = {70 : {'stress' : {5 : {'sz': {2: -5145.64404296875}, 'sy': {2: -5545.68701171875}, 'sx': {2: 6544.6015625}, 'szx': {2: -748.0105590820312}, 'sxy': {2: 6944.29248046875}, 'syz': {2: 1537.8612060546875}}}}}
        dd = {'sz': {2: -5145.64404296875}, 'sy': {2: -5545.68701171875}, 'sx': {2: 6544.6015625}, 'szx': {2: -748.0105590820312}, 'sxy': {2: 6944.29248046875}, 'syz': {2: 1537.8612060546875}}
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == dd)
        
        mod = {70 : {'stress' : {5 :{'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}}}}
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable('stress', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == modd)
        
        # Back to original
        self.mili.modify_state_variable('stress', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value == dd)

    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        d = {70 : {'stress' : {5 : {'sy': {2: -5545.68701171875}}}}}
        dd = -5545.68701171875
        
        # Before change
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == dd)
        
        # After change
        mod = {70 : {'stress' : {5 : {'sy': {2 : 12.0}}}}}
        modd = 12.0
        self.mili.modify_state_variable('stress[sy]', 'beam', mod, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == modd)
        
        # Back to original
        self.mili.modify_state_variable('stress[sy]', 'beam', d, 5, [70], [2])
        answer = self.mili.query('stress[sy]', 'beam', None, [5], [70], False, [2], False)
        assert(answer.state_answers[0].items[0].value['sy'][2] == dd)
    
if __name__ == '__main__':
    unittest.main()
