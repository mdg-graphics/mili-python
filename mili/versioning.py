"""
Copyright (c) 2016-2021, Lawrence Livermore National Security, LLC. 
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

'''
An attribute has a name and a value
e.g. name = 'mo_id' value = '5'
Always print is used in case the value is 0 for displaying in repr
'''
class Attribute:
    def __init__(self, name, value, always_print=False):
        self.name = name
        self.value = value
        self.always_print = always_print

    def __repr__(self):
        ret = ''
        if self.value or self.always_print:
            return self.name + ': ' + str(self.value) + '\n'
        return ret

'''
An item is an object that represents a component of an answer
Any number of its attributes may be None if they aren't filled
'''
class Item:
    def __init__(self, name=None, material=None, mo_id=None, label=None, class_name=None, modify=None, value=None):
        self.name = name
        self.material = material
        self.mo_id = mo_id
        self.label = label
        self.class_name = class_name
        self.modify = modify
        self.value = value
        self.always_print = False  # always print the value

    def set(self, value):
        self.value = value
        self.always_print = True

    def __str__(self):
        attributes = []
        attvalues = [self.name, self.material, self.mo_id, self.label, self.class_name, self.modify, self.value]
        attnames = ['name', 'material', 'mo_id', 'label', 'class_name', 'modify', 'value']

        for i in range(len(attvalues)):
            name, value = attnames[i], attvalues[i]
            if name == 'value' and self.always_print: attributes.append(Attribute(name, value, True))
            else: attributes.append(Attribute(name, value))

        ret = ''
        for attribute in attributes:
            ret += str(attribute)
        return ret + '\n'

'''
Each state in a query has one StateAnswer
Each contains a list of items and a state number
'''
class StateAnswer:
    def __init__(self):
        self.items = []
        self.state_number = None

    def __str__(self):
        ret = '\nstate number: ' + str(self.state_number) + '\n'
        for item in self.items:
            ret += str(item)
        return ret

class Answer:
    '''
    The encompassing object that is returned by a query.
    This object contains either:
    1. A list of items
    2. A list of state answers
    If the answer is state based (comes from a query with state_numbers),
    the answer contains a list of state answers.
    '''
    def __init__(self):
        self.state_answers = []
        self.items = []

    def __str__(self):
        ret = ''

        if not len(self.state_answers):
            for item in self.items:
                ret += str(item)
        else:
            for state in self.state_answers:
                ret += str(state)
        return ret

    def set(self, names, materials, mo_ids, labels, class_name, modify):
        num = 0
        if names: num = max(num, len(names))
        if materials: num = max(num, len(materials))
        if mo_ids: num = max(num, len(mo_ids))
        if labels: num = max(num, len(labels))

        for i in range(num):
            name = material = label = mo_id = None
            if names: name = names[i]
            if materials: material = materials[i]
            if labels: label = labels[i]
            if mo_ids: mo_id = mo_ids[i]
            if type(class_name) is list: name = class_name[i]
            else: name = class_name
            item = Item(name, material, mo_id, label, name, modify)
            self.items.append(item)
