import unittest
import read
'''
These are tests to assert the correctness of the Mili Reader

These tests use d3samp6.dyn
'''
from wheel.signatures import assertTrue
class TestMiliReader(unittest.TestCase):
    '''
    Set up the mili file object
    '''
    def setUp(self):
    	file_name = '../../Mili/MILI-toss_3_x86_64_ib-RZGENIE/d3samp6new.pltA'
        file_name_state_map = '../../Mili/MILI-toss_3_x86_64_ib-RZGENIE/d3samp6new.plt00'
    
        self.mili = read.Mili()
        self.mili.read(file_name, file_name_state_map)
        
    '''
    Testing whether the element numbers associated with a material are correct
    '''
    def test_element_number_material(self):
       answer = self.mili.elements_of_material('es_13')
       assert(len(answer.items) == 12)
       for i in range(len(answer.items)):
           assert(answer.items[i].mo_id == i + 1)

    
    '''
    Testing what nodes are associated with a material
    '''
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material("es_1", "beam")
        assert(len(answer.items) == 48)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == i + 1)
    
    '''
    Testing what nodes are associated with a label
    '''
    def test_nodes_label(self):
        answer = self.mili.nodes_of_elem(1, "brick")
        arr = [65, 81, 85, 69, 66, 82, 86, 70]
        assert(len(answer.items) == 8)
        for i in range(len(answer.items)):
           assert(answer.items[i].label == arr[i])
    
    '''
    Testing accessing a variable at a given state
    '''
    def test_state_varialble(self):
        answer = self.mili.query(['matke'], "mat", None, [1,2], [3, 4], None)
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
        answer = self.mili.query(['nodpos[ux]'], "node", None, [4], [3])
        assert(answer.state_answers[0].items[0].value == 0.9659258127212524)
        
        answer = self.mili.query(['ux'], "node", None, [4], [3])
        assert(answer.state_answers[0].items[0].value == 0.9659258127212524)
    
    '''
    Testing the accessing of a vector, in this case node position
    '''
    def test_state_variable_vector(self):
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3])
        arr = [0.9659258127212524, 0.4330127537250519, 1.000000238418579]
        assert(answer.state_answers[0].items[0].value == arr)
    
    '''
    Testing the modification of a scalar state variable
    '''
    def test_modify_state_variable(self):
        # Before change
        answer = self.mili.query(['matke'], 'mat', None, [1], [3])
        assert(answer.state_answers[0].items[0].value == 0.0)
        
        # After change
        self.mili.modify_state_variable(['matke'], "mat", [5.0], [1], [3])
        answer = self.mili.query(['matke'], 'mat', None, [1], [3])
        assert(answer.state_answers[0].items[0].value == 5.0)
        
        # Back to original
        self.mili.modify_state_variable(['matke'], "mat", [0.0], [1], [3])
        answer = self.mili.query(['matke'], 'mat', None, [1], [3])
        assert(answer.state_answers[0].items[0].value == 0.0)
    
    '''
    Testing the modification of a vector state variable
    '''
    def test_modify_vector(self):
        # Before change
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3])
        assert(answer.state_answers[0].items[0].value ==[0.9659258127212524, 0.4330127537250519, 1.000000238418579])
        
        
        # After change
        self.mili.modify_state_variable(['nodpos'], "node", [5.0, 6.0, 9.0], [4], [3])
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3])
        assert(answer.state_answers[0].items[0].value ==[5.0, 6.0, 9.0])
        
        
        # Back to original
        self.mili.modify_state_variable(['nodpos'], "node", [0.9659258127212524, 0.4330127537250519, 1.000000238418579], [4], [3])
        answer = self.mili.query(['nodpos'], 'node', None, [4], [3])
        assert(answer.state_answers[0].items[0].value ==[0.9659258127212524, 0.4330127537250519, 1.000000238418579])
        
    '''
    Test accessing a vector array
    '''
    def test_state_variable_vector_array(self):
        answer = self.mili.query(['stress'], "beam", None, [5], [70], False, [2])
        d = {'sz': {2: -5128.0888671875}, 'sy': {2: -5506.6328125}, 'sx': {2: 6499.97216796875}, 'szx': {2: -772.875}, 'sxy': {2: 6878.185546875}, 'syz': {2: 1544.0760498046875}}
        assert(answer.state_answers[0].items[0].value == d)
    
    '''
    Test accessing a vector array component
    '''
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query(['stress[sy]'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value == {'sy' : {2: -5506.6328125}})
        
    '''
    Test modifying a vector array 
    '''
    def test_modify_vector_array(self):
         # Before change
        d = [6499.97216796875, -5506.6328125, -5128.0888671875, 6878.185546875, 1544.0760498046875, -772.875]
        dd = {'sz': {2: -5128.0888671875}, 'sy': {2: -5506.6328125}, 'sx': {2: 6499.97216796875}, 'szx': {2: -772.875}, 'sxy': {2: 6878.185546875}, 'syz': {2: 1544.0760498046875}}
        answer = self.mili.query(['stress'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value == dd)
        
        mod = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        modd = {'sz': {2: 1.0}, 'sy': {2: 1.0}, 'sx': {2: 1.0}, 'szx': {2: 1.0}, 'sxy': {2: 1.0}, 'syz': {2: 1.0}}
        self.mili.modify_state_variable(['stress'], "beam", mod, [5], [70], [2])
        answer = self.mili.query(['stress'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value == modd)
        
        # Back to original
        self.mili.modify_state_variable(['stress'], "beam", d, [5], [70], [2])
        answer = self.mili.query(['stress'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value == dd)
    '''
    Test modifying a vector array component
    '''
    def test_modify_vector_array_component(self):
        # Before change
        answer = self.mili.query(['stress[sy]'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value['sy'][2] == -5506.6328125)
        
        # After change
        self.mili.modify_state_variable(['stress[sy]'], "beam", [12.0], [5], [70], [2])
        answer = self.mili.query(['stress[sy]'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value['sy'][2] == 12.0)
        
        # Back to original
        self.mili.modify_state_variable(['stress[sy]'], "beam", [-5506.6328125], [5], [70], [2])
        answer = self.mili.query(['stress[sy]'], "beam", None, [5], [70], False, [2])
        assert(answer.state_answers[0].items[0].value['sy'][2] == -5506.6328125)
    
if __name__ == '__main__':
    unittest.main()
