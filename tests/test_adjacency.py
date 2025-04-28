#!/usr/bin/env python3

"""
Testing for Adjacency Module.

SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili import reader, adjacency
from mili.mdg_defines import EntityType
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestSerialGeometricMeshInfo(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestSerialGeometricMeshInfo.file_name )

    def test_compute_centroid(self):
        """Test the compute_centroid function."""
        # Brick elements
        centroid = self.mili.geometry.compute_centroid("brick", label=22, state=1)
        np.testing.assert_allclose( centroid, np.array([0.939143, 0.939143, 2.333333]), rtol=6.0e-7 )

        centroid = self.mili.geometry.compute_centroid("brick", label=7, state=100)
        np.testing.assert_allclose( centroid, np.array([0.583133, 0.15625 , 2.272023]), rtol=2.0e-7 )

        # Shell elements
        centroid = self.mili.geometry.compute_centroid("shell", label=3, state=1)
        np.testing.assert_allclose( centroid, np.array([0.426883, 0.426883, 2.      ]), rtol=2.0e-7 )

        centroid = self.mili.geometry.compute_centroid("shell", label=4, state=100)
        np.testing.assert_allclose( centroid, np.array([0.597617, 0.597617, 1.845676]), rtol=8.0e-7 )

        # beam elements
        centroid = self.mili.geometry.compute_centroid("beam", label=33, state=1)
        np.testing.assert_allclose( centroid, np.array([0.579082, 0.334333, 1.466667]), rtol=1.0e-6 )

        centroid = self.mili.geometry.compute_centroid("beam", label=45, state=100)
        np.testing.assert_allclose( centroid, np.array([-2.934491e-08,  7.380999e-01,  1.125251e+00]) )

        # Nodes
        centroid = self.mili.geometry.compute_centroid("node", label=1, state=1)
        np.testing.assert_allclose( centroid, np.array([1.0, 0.0, 0.0]) )

        centroid = self.mili.geometry.compute_centroid("node", label=132, state=100)
        np.testing.assert_allclose( centroid, np.array([1.5, 0.0, 2.672029]) )

    def test_nodes_in_radius(self):
        """Test the nodes_in_radius function."""
        nodes = self.mili.geometry.nodes_within_radius([0.0,0.0,0.0], 1.0, 1)
        np.testing.assert_equal( nodes, [1, 3, 16] )

        nodes = self.mili.geometry.nodes_within_radius([1.006000, 0.0, 0.6], 0.5, 1)
        np.testing.assert_equal( nodes, [8, 17, 18, 19, 20] )

        nodes = self.mili.geometry.nodes_within_radius([1.082308, 0.0, 0.5467003], 0.4, 75)
        np.testing.assert_equal( nodes, [8, 17, 18, 19, 20] )

    def test_elems_of_nodes(self):
        """Test the elems_of_nodes function."""
        elems = self.mili.geometry.elems_of_nodes(120)
        np.testing.assert_equal( elems["brick"], [31,32,33,34] )

        elems = self.mili.geometry.elems_of_nodes(15)
        np.testing.assert_equal( elems["cseg"], [6,9] )
        np.testing.assert_equal( elems["beam"], [6, 42] )
        np.testing.assert_equal( elems["shell"], [6, 11] )

        elems = self.mili.geometry.elems_of_nodes(3)
        # beam 1, 2, 12
        np.testing.assert_equal( elems["beam"], [1,2,12] )

class TestSerialAdjacencyMapping(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( TestSerialAdjacencyMapping.file_name )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

    def test_compute_centroid(self):
        """Test the compute_centroid function."""
        # Brick elements
        centroid = self.adjacency.compute_centroid("brick", label=22, state=1)
        np.testing.assert_allclose( centroid, np.array([0.939143, 0.939143, 2.333333]), rtol=6.0e-7 )

        centroid = self.adjacency.compute_centroid(EntityType.BRICK, label=7, state=100)
        np.testing.assert_allclose( centroid, np.array([0.583133, 0.15625 , 2.272023]), rtol=2.0e-7 )

        # Shell elements
        centroid = self.adjacency.compute_centroid("shell", label=3, state=1)
        np.testing.assert_allclose( centroid, np.array([0.426883, 0.426883, 2.      ]), rtol=2.0e-7 )

        centroid = self.adjacency.compute_centroid(EntityType.SHELL, label=4, state=100)
        np.testing.assert_allclose( centroid, np.array([0.597617, 0.597617, 1.845676]), rtol=8.0e-7 )

        # beam elements
        centroid = self.adjacency.compute_centroid("beam", label=33, state=1)
        np.testing.assert_allclose( centroid, np.array([0.579082, 0.334333, 1.466667]), rtol=1.0e-6 )

        centroid = self.adjacency.compute_centroid(EntityType.BEAM, label=45, state=100)
        np.testing.assert_allclose( centroid, np.array([-2.934491e-08,  7.380999e-01,  1.125251e+00]) )

        # Nodes
        centroid = self.adjacency.compute_centroid("node", label=1, state=1)
        np.testing.assert_allclose( centroid, np.array([1.0, 0.0, 0.0]) )

        centroid = self.adjacency.compute_centroid(EntityType.NODE, label=132, state=100)
        np.testing.assert_allclose( centroid, np.array([1.5, 0.0, 2.672029]) )

    def test_mesh_entities_in_radius(self):
        """Test the mesh_entities_in_radius function."""
        elems = self.adjacency.mesh_entities_within_radius("node", 120, 1, 0.1)
        np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

        elems = self.adjacency.mesh_entities_within_radius(EntityType.BRICK, 32, 1, 0.4)
        np.testing.assert_equal( sorted(elems["brick"]), [25,26,27,28,31,32,33,34] )

        elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3)
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
        np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

        # Test limiting by material
        elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=1)
        self.assertEqual( list(elems.keys()), ["beam", "node"] )
        np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )

        elems = self.adjacency.mesh_entities_within_radius(EntityType.SHELL, 6, 1, 0.3, material=3)
        self.assertEqual( list(elems.keys()), ["shell", "node"] )
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

        elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=4)
        self.assertEqual( list(elems.keys()), ["cseg", "node"] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

        elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=[3,4])
        self.assertEqual( list(elems.keys()), ["shell", "cseg", "node"] )
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

    def test_mesh_entities_near_coordinate(self):
        """Test the mesh_entities_near_coordinate function."""
        elems = self.adjacency.mesh_entities_near_coordinate([1.0825318, 0.62499994, 3.], 1, 0.1)
        np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

        elems = self.adjacency.mesh_entities_near_coordinate([1.2828925, 0.34374997, 2.8666668], 1, 0.4)
        np.testing.assert_equal( sorted(elems["brick"]), [25,26,27,28,31,32,33,34] )

        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3)
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
        np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

        # Test limiting by material
        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=1)
        self.assertEqual( list(elems.keys()), ["beam", "node"] )
        np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )

        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=3)
        self.assertEqual( list(elems.keys()), ["shell", "node"] )
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=4)
        self.assertEqual( list(elems.keys()), ["cseg", "node"] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=[1,3,4])
        self.assertEqual( list(elems.keys()), ["beam", "shell", "cseg", "node"] )
        np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )
        np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
        np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

    def test_elems_of_nodes(self):
        """Test the elems_of_nodes function."""
        elems = self.adjacency.elems_of_nodes(120)
        self.assertEqual(list(elems.keys()), ["brick"])
        np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

        elems = self.adjacency.elems_of_nodes(15)
        self.assertEqual(list(elems.keys()), ["beam", "shell", "cseg"])
        np.testing.assert_equal( sorted(elems["cseg"]), [6,9] )
        np.testing.assert_equal( sorted(elems["beam"]), [6, 42] )
        np.testing.assert_equal( sorted(elems["shell"]), [6, 11] )

        elems = self.adjacency.elems_of_nodes([3,120])
        self.assertEqual(list(elems.keys()), ["beam","brick"])
        np.testing.assert_equal( sorted(elems["beam"]), [1,2,12] )
        np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

        node_labels = [ 12,  13,  33,  37,  49,  50,  51,  53,  54,  65,  66,  69,  70,
                        73,  81,  82,  85,  86,  97, 101]
        elems = self.adjacency.elems_of_nodes(node_labels, material=4)
        self.assertEqual(list(elems.keys()), ["cseg"])
        np.testing.assert_equal( sorted(elems["cseg"]), [1,2,3,4,5,7,8] )

        elems = self.adjacency.elems_of_nodes(node_labels, material=5)
        self.assertEqual(list(elems.keys()), ["cseg"])
        np.testing.assert_equal( sorted(elems["cseg"]), [13,14,15,16,17,19,20] )

        elems = self.adjacency.elems_of_nodes(node_labels, material=[4,5])
        self.assertEqual(list(elems.keys()), ["cseg"])
        np.testing.assert_equal( sorted(elems["cseg"]), [1,2,3,4,5,7,8,13,14,15,16,17,19,20] )

    def test_nearest_node(self):
        """Test the nearest_node function."""
        node, dist = self.adjacency.nearest_node([0.0,0.0,0.0], 1)
        self.assertEqual(node, 3)
        self.assertEqual(dist, 0.9999999865388264)

        node, dist = self.adjacency.nearest_node([0.5,0.5,0.5], 44)
        self.assertEqual(node, 26)
        self.assertEqual(dist, 0.4160180349843221)

        node, dist = self.adjacency.nearest_node([0.75,1.5,2.8], 101)
        self.assertEqual(node, 140)
        self.assertEqual(dist, 0.23785513406115558)

        node, dist = self.adjacency.nearest_node([0.0,0.0,0.0], 1, material=[2,3])
        self.assertEqual(node, 49)
        self.assertEqual(dist, 2.0615528128088303)


    def test_nearest_element(self):
        """Test the nearest_element function."""
        class_name, label, dist = self.adjacency.nearest_element([0.0,0.0,0.0], 1)
        self.assertEqual(class_name, "beam")
        self.assertEqual(label, 12)
        self.assertEqual(dist, 0.778031964849994)

        class_name, label, dist = self.adjacency.nearest_element([0.5,0.5,0.5], 44)
        self.assertEqual(class_name, "beam")
        self.assertEqual(label, 13)
        self.assertEqual(dist, 0.1837874009986171)

        class_name, label, dist = self.adjacency.nearest_element([0.75,1.5,2.8], 101)
        self.assertEqual(class_name, "brick")
        self.assertEqual(label, 36)
        self.assertEqual(dist, 0.5292167548771606)

        class_name, label, dist = self.adjacency.nearest_element([0.0,0.0,0.0], 1, material=[3])
        self.assertEqual(class_name, "shell")
        self.assertEqual(label, 1)
        self.assertEqual(dist, 2.089128544941847)

    def test_neighbor_elements(self):
        """Test the neighbor_elements funcion."""
        elems = self.adjacency.neighbor_elements("brick", 1)
        self.assertEqual(list(elems.keys()), ["brick", "cseg"])
        np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 7, 8, 9, 10])
        np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 16, 17])

        elems = self.adjacency.neighbor_elements(EntityType.BRICK, 1, material = 2)
        self.assertEqual(list(elems.keys()), ["brick"])
        np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 7, 8, 9, 10])

        elems = self.adjacency.neighbor_elements("brick", 1, material = 5)
        self.assertEqual(list(elems.keys()), ["cseg"])
        np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 16, 17])

        elems = self.adjacency.neighbor_elements("brick", 1, neighbor_radius = 2)
        self.assertEqual(list(elems.keys()), ["brick", "cseg"])
        np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                 11, 12, 13, 14, 15, 16, 17, 18,
                                                 19, 21, 23, 25, 27, 29, 31, 33, 35])
        np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 15, 16, 17, 18, 19, 20, 21])

        elems = self.adjacency.neighbor_elements(EntityType.BRICK, 1, material=[2,3], neighbor_radius = 3)
        self.assertEqual(list(elems.keys()), ["brick"])
        np.testing.assert_equal(sorted(elems["brick"]), np.arange(1, 37)) # All brick elements are within 3 neighbors of brick 1


class ParallelAdjacencyTests:
    class TestParallelGeometricMeshInfo(unittest.TestCase):
        def test_compute_centroid(self):
            """Test the compute_centroid function."""
            # Brick elements
            centroid = self.mili.geometry.compute_centroid("brick", label=22, state=1)
            np.testing.assert_allclose( centroid[5], np.array([0.939143, 0.939143, 2.333333]), rtol=6.0e-7 )

            centroid = self.mili.geometry.compute_centroid("brick", label=7, state=100)
            np.testing.assert_allclose( centroid[0], np.array([0.583133, 0.15625 , 2.272023]), rtol=2.0e-7 )

            # Shell elements
            centroid = self.mili.geometry.compute_centroid("shell", label=3, state=1)
            np.testing.assert_allclose( centroid[3], np.array([0.426883, 0.426883, 2.      ]), rtol=2.0e-7 )

            centroid = self.mili.geometry.compute_centroid("shell", label=4, state=100)
            np.testing.assert_allclose( centroid[3], np.array([0.597617, 0.597617, 1.845676]), rtol=8.0e-7 )

            # beam elements
            centroid = self.mili.geometry.compute_centroid("beam", label=33, state=1)
            np.testing.assert_allclose( centroid[4], np.array([0.579082, 0.334333, 1.466667]), rtol=1.0e-6 )

            centroid = self.mili.geometry.compute_centroid("beam", label=45, state=100)
            np.testing.assert_allclose( centroid[7], np.array([-2.934491e-08,  7.380999e-01,  1.125251e+00]) )

            # Nodes
            centroid = self.mili.geometry.compute_centroid("node", label=1, state=1)
            np.testing.assert_allclose( centroid[5], np.array([1.0, 0.0, 0.0]) )
            np.testing.assert_allclose( centroid[7], np.array([1.0, 0.0, 0.0]) )

            centroid = self.mili.geometry.compute_centroid("node", label=132, state=100)
            np.testing.assert_allclose( centroid[1], np.array([1.5, 0.0, 2.672029]) )

        def test_nodes_in_radius(self):
            """Test the nodes_in_radius function."""
            nodes = self.mili.geometry.nodes_within_radius([0.0,0.0,0.0], 1.0, 1)
            np.testing.assert_equal( np.unique(np.concatenate(nodes)), [1, 3, 16] )

            nodes = self.mili.geometry.nodes_within_radius([1.006000, 0.0, 0.6], 0.5, 1)
            np.testing.assert_equal( np.unique(np.concatenate(nodes)), [8, 17, 18, 19, 20] )

            nodes = self.mili.geometry.nodes_within_radius([1.082308, 0.0, 0.5467003], 0.4, 75)
            np.testing.assert_equal( np.unique(np.concatenate(nodes)), [8, 17, 18, 19, 20] )

        def test_elems_of_nodes(self):
            """Test the elems_of_nodes function."""
            elems = self.mili.geometry.elems_of_nodes(120)
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [31,32,33,34] )

            elems = self.mili.geometry.elems_of_nodes(15)
            cseg_elems = np.unique(np.concatenate([e.get("cseg",[]) for e in elems]))
            beam_elems = np.unique(np.concatenate([e.get("beam",[]) for e in elems]))
            shell_elems = np.unique(np.concatenate([e.get("shell",[]) for e in elems]))
            np.testing.assert_equal( cseg_elems, [6,9] )
            np.testing.assert_equal( beam_elems, [6, 42] )
            np.testing.assert_equal( shell_elems, [6, 11] )

            elems = self.mili.geometry.elems_of_nodes([3,120])
            beam_elems = np.unique(np.concatenate([e.get("beam",[]) for e in elems]))
            np.testing.assert_equal( beam_elems, [1,2,12] )
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [31,32,33,34] )

    class TestParallelAdjacencyMapping(unittest.TestCase):
        def test_elems_of_nodes(self):
            """Test the elems_of_nodes function."""
            elems = self.adjacency.elems_of_nodes(120)
            self.assertEqual(list(elems.keys()), ["brick"])
            np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

            elems = self.adjacency.elems_of_nodes(15)
            self.assertEqual(list(elems.keys()), ["beam", "shell", "cseg"])
            np.testing.assert_equal( sorted(elems["cseg"]), [6,9] )
            np.testing.assert_equal( sorted(elems["beam"]), [6, 42] )
            np.testing.assert_equal( sorted(elems["shell"]), [6, 11] )

            elems = self.adjacency.elems_of_nodes([3,120])
            self.assertEqual(list(elems.keys()), ["brick","beam"])
            np.testing.assert_equal( sorted(elems["beam"]), [1,2,12] )
            np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

            node_labels = [ 12,  13,  33,  37,  49,  50,  51,  53,  54,  65,  66,  69,  70,
                            73,  81,  82,  85,  86,  97, 101]
            elems = self.adjacency.elems_of_nodes(node_labels, material=4)
            self.assertEqual(list(elems.keys()), ["cseg"])
            np.testing.assert_equal( sorted(elems["cseg"]), [1,2,3,4,5,7,8] )

            elems = self.adjacency.elems_of_nodes(node_labels, material=5)
            self.assertEqual(list(elems.keys()), ["cseg"])
            np.testing.assert_equal( sorted(elems["cseg"]), [13,14,15,16,17,19,20] )

            elems = self.adjacency.elems_of_nodes(node_labels, material=[4,5])
            self.assertEqual(list(elems.keys()), ["cseg"])
            np.testing.assert_equal( sorted(elems["cseg"]), [1,2,3,4,5,7,8,13,14,15,16,17,19,20] )

        def test_mesh_entities_in_radius(self):
            """Test the mesh_entities_in_radius function."""
            elems = self.adjacency.mesh_entities_within_radius("node", 120, 1, 0.1)
            np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

            elems = self.adjacency.mesh_entities_within_radius("brick", 32, 1, 0.4)
            np.testing.assert_equal( sorted(elems["brick"]), [25,26,27,28,31,32,33,34] )

            elems = self.adjacency.mesh_entities_within_radius(EntityType.SHELL, 6, 1, 0.3)
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
            np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

            # Test limiting by material
            elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=1)
            self.assertEqual( list(elems.keys()), ["beam", "node"] )
            np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )

            elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=3)
            self.assertEqual( list(elems.keys()), ["shell", "node"] )
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

            elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3, material=4)
            self.assertEqual( list(elems.keys()), ["cseg", "node"] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

            elems = self.adjacency.mesh_entities_within_radius(EntityType.SHELL, 6, 1, 0.3, material=[3,4])
            self.assertEqual( list(elems.keys()), ["shell", "cseg", "node"] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

        def test_mesh_entities_near_coordinate(self):
            """Test the mesh_entities_near_coordinate function."""
            elems = self.adjacency.mesh_entities_near_coordinate([1.0825318, 0.62499994, 3.], 1, 0.1)
            np.testing.assert_equal( sorted(elems["brick"]), [31,32,33,34] )

            elems = self.adjacency.mesh_entities_near_coordinate([1.2828925, 0.34374997, 2.8666668], 1, 0.4)
            np.testing.assert_equal( sorted(elems["brick"]), [25,26,27,28,31,32,33,34] )

            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3)
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )
            np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

            # Test limiting by material
            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=1)
            self.assertEqual( list(elems.keys()), ["beam", "node"] )
            np.testing.assert_equal( sorted(elems["beam"]), [5,6,37,42] )

            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=3)
            self.assertEqual( list(elems.keys()), ["shell", "node"] )
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=4)
            self.assertEqual( list(elems.keys()), ["cseg", "node"] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )

            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3, material=[3,4])
            self.assertEqual( list(elems.keys()), ["shell", "cseg", "node"] )
            np.testing.assert_equal( sorted(elems["cseg"]), [2,3,5,6,8,9] )
            np.testing.assert_equal( sorted(elems["shell"]), [3,4,5,6,9,11] )

        def test_nearest_node(self):
            """Test the nearest_node function."""
            node, dist = self.adjacency.nearest_node([0.0,0.0,0.0], 1)
            self.assertEqual(node, 3)
            self.assertEqual(dist, 0.9999999865388264)

            node, dist = self.adjacency.nearest_node([0.5,0.5,0.5], 44)
            self.assertEqual(node, 26)
            self.assertEqual(dist, 0.4160180349843221)

            node, dist = self.adjacency.nearest_node([0.75,1.5,2.8], 101)
            self.assertEqual(node, 140)
            self.assertEqual(dist, 0.23785513406115558)

            node, dist = self.adjacency.nearest_node([0.0,0.0,0.0], 1, material=[2,3])
            self.assertEqual(node, 49)
            self.assertEqual(dist, 2.0615528128088303)

        def test_nearest_element(self):
            """Test the nearest_element function."""
            class_name, label, dist = self.adjacency.nearest_element([0.0,0.0,0.0], 1)
            self.assertEqual(class_name, "beam")
            self.assertEqual(label, 12)
            self.assertEqual(dist, 0.778031964849994)

            class_name, label, dist = self.adjacency.nearest_element([0.5,0.5,0.5], 44)
            self.assertEqual(class_name, "beam")
            self.assertEqual(label, 13)
            self.assertEqual(dist, 0.1837874009986171)

            class_name, label, dist = self.adjacency.nearest_element([0.75,1.5,2.8], 101)
            self.assertEqual(class_name, "brick")
            self.assertEqual(label, 36)
            self.assertEqual(dist, 0.5292167548771606)

            class_name, label, dist = self.adjacency.nearest_element([0.0,0.0,0.0], 1, material=[3])
            self.assertEqual(class_name, "shell")
            self.assertEqual(label, 1)
            self.assertEqual(dist, 2.089128544941847)

        def test_neighbor_elements(self):
            """Test the neighbor_elements funcion."""
            elems = self.adjacency.neighbor_elements("brick", 1)
            self.assertEqual(list(elems.keys()), ["brick", "cseg"])
            np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 7, 8, 9, 10])
            np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 16, 17])

            elems = self.adjacency.neighbor_elements("brick", 1, material = 2)
            self.assertEqual(list(elems.keys()), ["brick"])
            np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 7, 8, 9, 10])

            elems = self.adjacency.neighbor_elements("brick", 1, material = 5)
            self.assertEqual(list(elems.keys()), ["cseg"])
            np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 16, 17])

            elems = self.adjacency.neighbor_elements(EntityType.BRICK, 1, neighbor_radius = 2)
            self.assertEqual(list(elems.keys()), ["brick", "cseg"])
            np.testing.assert_equal(sorted(elems["brick"]), [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                    11, 12, 13, 14, 15, 16, 17, 18,
                                                    19, 21, 23, 25, 27, 29, 31, 33, 35])
            np.testing.assert_equal(sorted(elems["cseg"]), [13, 14, 15, 16, 17, 18, 19, 20, 21])

            elems = self.adjacency.neighbor_elements("brick", 1, material=2, neighbor_radius = 3)
            self.assertEqual(list(elems.keys()), ["brick"])
            np.testing.assert_equal(sorted(elems["brick"]), np.arange(1, 37)) # All brick elements are within 3 neighbors of brick 1

            elems = self.adjacency.neighbor_elements("brick", 1, material=[2,3], neighbor_radius = 3)
            self.assertEqual(list(elems.keys()), ["brick"])
            np.testing.assert_equal(sorted(elems["brick"]), np.arange(1, 37)) # All brick elements are within 3 neighbors of brick 1

###############################################
# Tests for each parallel wrapper class with and without merge_results=True
###############################################

class LoopWrapperGeometricMeshInfoTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperGeometricMeshInfoTests.file_name, suppress_parallel=True, merge_results=False )

class LoopWrapperAdjacencyMappingTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperAdjacencyMappingTests.file_name, suppress_parallel=True, merge_results=False )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

class ServerWrapperGeometricMeshInfoTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperGeometricMeshInfoTests.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

class ServerWrapperAdjacencyMappingTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperAdjacencyMappingTests.file_name, suppress_parallel=False, merge_results=False )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

    def tearDown(self):
        self.mili.close()

class LoopWrapperGeometricMeshInfoMergeResultsTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperGeometricMeshInfoTests.file_name, suppress_parallel=True, merge_results=True )

class LoopWrapperAdjacencyMappingMergeResultsTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperAdjacencyMappingTests.file_name, suppress_parallel=True, merge_results=True )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

class ServerWrapperGeometricMeshInfoMergeResultsTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperGeometricMeshInfoTests.file_name, suppress_parallel=False, merge_results=True )

    def tearDown(self):
        self.mili.close()

class ServerWrapperAdjacencyMappingMergeResultsTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperAdjacencyMappingTests.file_name, suppress_parallel=False, merge_results=True )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

    def tearDown(self):
        self.mili.close()