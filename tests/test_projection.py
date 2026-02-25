#!/usr/bin/env python3
"""
Testing for the MiliDatabase.query projection routines.

SPDX-License-Identifier: (MIT)
"""

import os
import unittest
from mili import reader
from mili.projection import (hex_to_nodal, quad_to_nodal, beam_to_nodal, tet_to_nodal,
                              particle_to_nodal)
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestProjectionRoutines(unittest.TestCase):
    """Testing for Projection routines."""

    def test_hex_to_nodal(self):
        """Test projecting result from hexes to nodes."""
        file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
        mili = reader.open_database( file_name )

        hex_result = mili.query("sx", "brick", states=[40,41], labels=[31,32])
        node_result = hex_to_nodal(mili, hex_result['sx'])

        self.assertEqual( node_result['class_name'], 'node')
        self.assertEqual( node_result['modifier'], '')
        self.assertEqual( node_result['title'], 'X Stress')
        self.assertEqual( node_result['source'], 'primal')
        np.testing.assert_equal( node_result['layout']['labels'], [99,100,103,104,115,116,119,120,131,132,135,136])
        np.testing.assert_equal( node_result['layout']['states'], [40,41])
        np.testing.assert_allclose( node_result['layout']['times'], [0.00039, 0.00040], rtol=3.0e-08)
        np.testing.assert_equal( node_result['layout']['components'], ['sx'])

        np.testing.assert_allclose(node_result['data'][0,0,0],  -94.135185)  # Node 99
        np.testing.assert_allclose(node_result['data'][0,1,0],  -94.135185)  # Node 100
        np.testing.assert_allclose(node_result['data'][0,2,0],  -94.135185)  # Node 103
        np.testing.assert_allclose(node_result['data'][0,3,0],  -94.135185)  # Node 104
        np.testing.assert_allclose(node_result['data'][0,4,0],  -261.15353)  # Node 115
        np.testing.assert_allclose(node_result['data'][0,5,0],  -261.15353)  # Node 116
        np.testing.assert_allclose(node_result['data'][0,6,0],  -261.15353)  # Node 119
        np.testing.assert_allclose(node_result['data'][0,7,0],  -261.15353)  # Node 120
        np.testing.assert_allclose(node_result['data'][0,8,0],  -397.80515)  # Node 131
        np.testing.assert_allclose(node_result['data'][0,9,0],  -397.80515)  # Node 132
        np.testing.assert_allclose(node_result['data'][0,10,0], -397.80515)  # Node 135
        np.testing.assert_allclose(node_result['data'][0,11,0], -397.80515)  # Node 136

        np.testing.assert_allclose(node_result['data'][1,0,0],  -440.79117)  # Node 99
        np.testing.assert_allclose(node_result['data'][1,1,0],  -440.79117)  # Node 100
        np.testing.assert_allclose(node_result['data'][1,2,0],  -440.79117)  # Node 103
        np.testing.assert_allclose(node_result['data'][1,3,0],  -440.79117)  # Node 104
        np.testing.assert_allclose(node_result['data'][1,4,0],  -536.29926)  # Node 115
        np.testing.assert_allclose(node_result['data'][1,5,0],  -536.29926)  # Node 116
        np.testing.assert_allclose(node_result['data'][1,6,0],  -536.29926)  # Node 119
        np.testing.assert_allclose(node_result['data'][1,7,0],  -536.29926)  # Node 120
        np.testing.assert_allclose(node_result['data'][1,8,0],  -614.4423)   # Node 131
        np.testing.assert_allclose(node_result['data'][1,9,0],  -614.4423)   # Node 132
        np.testing.assert_allclose(node_result['data'][1,10,0], -614.4423)   # Node 135
        np.testing.assert_allclose(node_result['data'][1,11,0], -614.4423)   # Node 136

    def test_quad_to_nodal(self):
        """Test projecting result from quads to nodes."""
        file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
        mili = reader.open_database( file_name )

        quad_result = mili.query("sy", "shell", states=[40,41], labels=[1,2])
        node_result = quad_to_nodal(mili, quad_result['sy'])

        self.assertEqual( node_result['class_name'], 'node')
        self.assertEqual( node_result['modifier'], '')
        self.assertEqual( node_result['title'], 'Y Stress')
        self.assertEqual( node_result['source'], 'primal')
        np.testing.assert_equal( node_result['layout']['labels'], [12,13,49,50,53,54])
        np.testing.assert_equal( node_result['layout']['states'], [40,41])
        np.testing.assert_allclose( node_result['layout']['times'], [0.00039, 0.00040], rtol=3.0e-08)
        np.testing.assert_equal( node_result['layout']['components'], ['sy ipt. 1', 'sy ipt. 2'])

        np.testing.assert_allclose(node_result['data'][0,0,:], [ 9011.6045, -5640.1865])  # Node 12
        np.testing.assert_allclose(node_result['data'][0,1,:], [ 9011.6045, -5640.1865])  # Node 13
        np.testing.assert_allclose(node_result['data'][0,2,:], [ 8707.642 , -4608.2295])  # Node 49
        np.testing.assert_allclose(node_result['data'][0,3,:], [ 8707.642 , -4608.2295])  # Node 50
        np.testing.assert_allclose(node_result['data'][0,4,:], [ 8859.623 , -5124.208 ])  # Node 53
        np.testing.assert_allclose(node_result['data'][0,5,:], [ 8859.623 , -5124.208 ])  # Node 54

        np.testing.assert_allclose(node_result['data'][1,0,:], [ 5136.911 , -3704.259 ])  # Node 12
        np.testing.assert_allclose(node_result['data'][1,1,:], [ 5136.911 , -3704.259 ])  # Node 13
        np.testing.assert_allclose(node_result['data'][1,2,:], [ 7967.586 , -6283.525 ])  # Node 49
        np.testing.assert_allclose(node_result['data'][1,3,:], [ 7967.586 , -6283.525 ])  # Node 50
        np.testing.assert_allclose(node_result['data'][1,4,:], [ 6552.2485, -4993.892 ])  # Node 53
        np.testing.assert_allclose(node_result['data'][1,5,:], [ 6552.2485, -4993.892 ])  # Node 54

    def test_beam_to_nodal(self):
        """Test projecting result from beams to nodes."""
        file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')
        mili = reader.open_database( file_name )

        beam_result = mili.query("sz", "beam", states=[40,41], labels=[10,11])
        node_result = beam_to_nodal(mili, beam_result['sz'])

        self.assertEqual( node_result['class_name'], 'node')
        self.assertEqual( node_result['modifier'], '')
        self.assertEqual( node_result['title'], 'Z Stress')
        self.assertEqual( node_result['source'], 'primal')
        np.testing.assert_equal( node_result['layout']['labels'], [8,19,20])
        np.testing.assert_equal( node_result['layout']['states'], [40,41])
        np.testing.assert_allclose( node_result['layout']['times'], [0.00039, 0.00040], rtol=3.0e-08)
        np.testing.assert_equal( node_result['layout']['components'], ['sz ipt. 1', 'sz ipt. 2', 'sz ipt. 3', 'sz ipt. 4'])

        np.testing.assert_allclose(node_result['data'][0,0,:], [ -6881.7715, -53600.758 , -53600.758 ,  -6881.7715])  # Node 8
        np.testing.assert_allclose(node_result['data'][0,1,:], [-21330.164 , -45180.36  , -45180.36  , -21330.164 ])  # Node 20
        np.testing.assert_allclose(node_result['data'][0,2,:], [-14105.968 , -49390.56  , -49390.56  , -14105.968 ])  # Node 36

        np.testing.assert_allclose(node_result['data'][1,0,:], [ -6650.4854, -53988.1   , -53988.1   ,  -6650.4854])  # Node 8
        np.testing.assert_allclose(node_result['data'][1,1,:], [-17625.666 , -40216.64  , -40216.64  , -17625.666 ])  # Node 20
        np.testing.assert_allclose(node_result['data'][1,2,:], [-12138.076 , -47102.37  , -47102.37  , -12138.076 ])  # Node 36

    def test_tet_to_nodal(self):
        """Test projecting result from tets to nodes."""
        file_name = os.path.join(dir_path,'data','serial','tet','tet1_t4.plt')
        mili = reader.open_database( file_name )

        tet_result = mili.query("sx", "tet", states=[74,75], labels=[1,2])
        node_result = tet_to_nodal(mili, tet_result['sx'])

        self.assertEqual( node_result['class_name'], 'node')
        self.assertEqual( node_result['modifier'], '')
        self.assertEqual( node_result['title'], 'X Stress')
        self.assertEqual( node_result['source'], 'primal')
        np.testing.assert_equal( node_result['layout']['labels'], [619, 620, 842, 843, 844, 845, 846])
        np.testing.assert_equal( node_result['layout']['states'], [74,75])
        np.testing.assert_allclose( node_result['layout']['times'], [73.0,74.0], rtol=3.0e-08)
        np.testing.assert_equal( node_result['layout']['components'], ['sx'])

        np.testing.assert_allclose(node_result['data'][0,0,0],  [ 0.013191545], rtol=4.0e-07)  # Node 619
        np.testing.assert_allclose(node_result['data'][0,1,0],  [ 0.013191545], rtol=4.0e-07)  # Node 620
        np.testing.assert_allclose(node_result['data'][0,2,0],  [-0.005808995], rtol=1.3e-06)  # Node 842
        np.testing.assert_allclose(node_result['data'][0,3,0],  [ 0.013191545], rtol=4.0e-07)  # Node 843
        np.testing.assert_allclose(node_result['data'][0,4,0],  [-0.018900883])  # Node 844
        np.testing.assert_allclose(node_result['data'][0,5,0],  [-0.018900883])  # Node 845
        np.testing.assert_allclose(node_result['data'][0,6,0],  [-0.018900883])  # Node 846

        np.testing.assert_allclose(node_result['data'][1,0,0],  [ 0.01314164], rtol=1.1e-07)  # Node 619
        np.testing.assert_allclose(node_result['data'][1,1,0],  [ 0.01314164], rtol=1.1e-07)  # Node 620
        np.testing.assert_allclose(node_result['data'][1,2,0],  [-0.00585147], rtol=2.3e-07)  # Node 842
        np.testing.assert_allclose(node_result['data'][1,3,0],  [ 0.01314164], rtol=1.1e-07)  # Node 843
        np.testing.assert_allclose(node_result['data'][1,4,0],  [-0.01893815], rtol=1.2e-07)  # Node 844
        np.testing.assert_allclose(node_result['data'][1,5,0],  [-0.01893815], rtol=1.2e-07)  # Node 845
        np.testing.assert_allclose(node_result['data'][1,6,0],  [-0.01893815], rtol=1.2e-07)  # Node 846


    def test_particle_to_nodal(self):
        """Test projecting result from particles to nodes."""
        file_name = os.path.join(dir_path,'data','serial','dbl_nodtang','dblplt')
        mili = reader.open_database( file_name )

        particle_result = mili.query("nodtangmag", "cbs1_particle", states=[40,41])
        node_result = particle_to_nodal(mili, particle_result['nodtangmag'])

        self.assertEqual( node_result['class_name'], 'node')
        self.assertEqual( node_result['modifier'], '')
        self.assertEqual( node_result['title'], 'Nodal Tangential Traction Magnitude')
        self.assertEqual( node_result['source'], 'derived')
        np.testing.assert_equal( node_result['layout']['labels'], [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,125])
        np.testing.assert_equal( node_result['layout']['states'], [40,41])
        np.testing.assert_allclose( node_result['layout']['times'], [3.9, 4.0], rtol=3.0e-08)
        np.testing.assert_equal( node_result['layout']['components'], ['nodtangmag'])

        np.testing.assert_allclose(node_result['data'], particle_result['nodtangmag']['data'])