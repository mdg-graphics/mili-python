#!/usr/bin/env python3

"""
Copyright (c) 2016-present, Lawrence Livermore National Security, LLC.
 Produced at the Lawrence Livermore National Laboratory. Written by:

 William Tobin (tobin6@llnl.gov),
 Ryan Hathaway (hathaway6@llnl.gov),
 Kevin Durrenberger (durrenberger1@llnl.gov).

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
from mili import reader, adjacency
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

    def test_mesh_entities_in_radius(self):
        """Test the mesh_entities_in_radius function."""
        elems = self.adjacency.mesh_entities_within_radius("node", 120, 1, 0.1)
        np.testing.assert_equal( elems["brick"], [31,32,33,34] )

        elems = self.adjacency.mesh_entities_within_radius("brick", 32, 1, 0.4)
        np.testing.assert_equal( elems["brick"], [25,26,27,28,31,32,33,34] )

        elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.3)
        np.testing.assert_equal( elems["shell"], [3,4,5,6,9,11] )
        np.testing.assert_equal( elems["beam"], [5,6,37,42] )
        np.testing.assert_equal( elems["cseg"], [2,3,5,6,8,9] )

    def test_mesh_entities_near_coordinate(self):
        """Test the mesh_entities_near_coordinate function."""
        elems = self.adjacency.mesh_entities_near_coordinate([1.0825318, 0.62499994, 3.], 1, 0.1)
        np.testing.assert_equal( elems["brick"], [31,32,33,34] )

        elems = self.adjacency.mesh_entities_near_coordinate([1.2828925, 0.34374997, 2.8666668], 1, 0.4)
        np.testing.assert_equal( elems["brick"], [25,26,27,28,31,32,33,34] )

        elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3)
        np.testing.assert_equal( elems["shell"], [3,4,5,6,9,11] )
        np.testing.assert_equal( elems["beam"], [5,6,37,42] )
        np.testing.assert_equal( elems["cseg"], [2,3,5,6,8,9] )

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
        def test_mesh_entities_in_radius(self):
            """Test the mesh_entities_in_radius function."""
            elems = self.adjacency.mesh_entities_within_radius("node", 120, 1, 0.1)
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [31,32,33,34] )

            elems = self.adjacency.mesh_entities_within_radius("brick", 32, 1, 0.4)
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [25,26,27,28,31,32,33,34] )

            elems = self.adjacency.mesh_entities_within_radius("shell", 6, 1, 0.29)
            cseg_elems = np.unique(np.concatenate([e.get("cseg",[]) for e in elems]))
            beam_elems = np.unique(np.concatenate([e.get("beam",[]) for e in elems]))
            shell_elems = np.unique(np.concatenate([e.get("shell",[]) for e in elems]))
            np.testing.assert_equal( shell_elems, [3,4,5,6,9,11] )
            np.testing.assert_equal( beam_elems, [5,6,37,42] )
            np.testing.assert_equal( cseg_elems, [2,3,5,6,8,9] )

        def test_mesh_entities_near_coordinate(self):
            """Test the mesh_entities_near_coordinate function."""
            elems = self.adjacency.mesh_entities_near_coordinate([1.0825318, 0.62499994, 3.], 1, 0.1)
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [31,32,33,34] )

            elems = self.adjacency.mesh_entities_near_coordinate([1.2828925, 0.34374997, 2.8666668], 1, 0.4)
            brick_elems = np.unique(np.concatenate([e.get("brick",[]) for e in elems]))
            np.testing.assert_equal( brick_elems, [25,26,27,28,31,32,33,34] )

            elems = self.adjacency.mesh_entities_near_coordinate([0.21874996, 0.8163861, 2.], 1, 0.3)
            cseg_elems = np.unique(np.concatenate([e.get("cseg",[]) for e in elems]))
            beam_elems = np.unique(np.concatenate([e.get("beam",[]) for e in elems]))
            shell_elems = np.unique(np.concatenate([e.get("shell",[]) for e in elems]))
            np.testing.assert_equal( shell_elems, [3,4,5,6,9,11] )
            np.testing.assert_equal( beam_elems, [5,6,37,42] )
            np.testing.assert_equal( cseg_elems, [2,3,5,6,8,9] )

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

###############################################
# Tests for each parallel wrapper class
###############################################

class LoopWrapperGeometricMeshInfoTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperGeometricMeshInfoTests.file_name, suppress_parallel=True )

class LoopWrapperAdjacencyMappingTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperAdjacencyMappingTests.file_name, suppress_parallel=True )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

class PoolWrapperGeometricMeshInfoTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( PoolWrapperGeometricMeshInfoTests.file_name, suppress_parallel=False )

class PoolWrapperAdjacencyMappingTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( PoolWrapperAdjacencyMappingTests.file_name, suppress_parallel=False )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)

class ServerWrapperGeometricMeshInfoTests(ParallelAdjacencyTests.TestParallelGeometricMeshInfo):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperGeometricMeshInfoTests.file_name, suppress_parallel=False, experimental=True )

class ServerWrapperAdjacencyMappingTests(ParallelAdjacencyTests.TestParallelAdjacencyMapping):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperAdjacencyMappingTests.file_name, suppress_parallel=False, experimental=True )
        self.adjacency = adjacency.AdjacencyMapping(self.mili)