#!/usr/bin/env python3
"""
Testing for the Derived Variables module.

SPDX-License-Identifier: (MIT)
"""

import re
import os
import unittest
from mili.reader import open_database, combine
from mili.milidatabase import MiliPythonError
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class SerialDerivedExpressions(unittest.TestCase):
    """Test Serial implementation of Derived Expressions Wrapper class."""
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = open_database( SerialDerivedExpressions.file_name, suppress_parallel = True )

    def test_supported_variables(self):
        EXPECTED = ['disp_x', 'disp_y', 'disp_z', 'disp_mag', 'disp_rad_mag_xy',
                    'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'vol_strain',
                    'prin_strain1', 'prin_strain2', 'prin_strain3',
                    'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3',
                    'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt',
                    'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt',
                    'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure',
                    'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3', 'max_shear_stress',
                    'triaxiality', 'norm_press', 'eps_rate', 'nodtangmag', 'mat_cog_disp_x', 'mat_cog_disp_y',
                    'mat_cog_disp_z', 'element_volume', 'area', 'centroid', 'surfstrainx', 'surfstrainy',
                    'surfstrainz', 'surfstrainxy', 'surfstrainyz', 'surfstrainzx',
                    ]
        supported_variables = self.mili.supported_derived_variables()
        self.assertEqual( EXPECTED, supported_variables )

    def test_derived_variables_of_class(self):
        BRICK_DERIVED = ['vol_strain', 'prin_strain1', 'prin_strain2', 'prin_strain3', 'prin_dev_strain1', 'prin_dev_strain2',
                         'prin_dev_strain3', 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt', 'prin_dev_strain1_alt',
                         'prin_dev_strain2_alt', 'prin_dev_strain3_alt', 'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress',
                         'pressure', 'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3', 'max_shear_stress', 'triaxiality', 'norm_press',
                         'element_volume', 'centroid', 'surfstrainx', 'surfstrainy', 'surfstrainz', 'surfstrainxy', 'surfstrainyz', 'surfstrainzx']
        BEAM_DERIVED = ['prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure', 'prin_dev_stress1', 'prin_dev_stress2',
                        'prin_dev_stress3', 'max_shear_stress', 'triaxiality', 'norm_press', 'eps_rate', 'centroid']
        SHELL_DERIVED = ['vol_strain', 'prin_strain1', 'prin_strain2', 'prin_strain3', 'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3',
                         'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt', 'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt',
                         'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure', 'prin_dev_stress1', 'prin_dev_stress2',
                         'prin_dev_stress3', 'max_shear_stress', 'triaxiality', 'norm_press', 'area', 'centroid']
        CSEG_DERIVED = ['area', 'centroid']
        NODE_DERIVED = ['disp_x', 'disp_y', 'disp_z', 'disp_mag', 'disp_rad_mag_xy', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'centroid']

        self.assertEqual( self.mili.derived_variables_of_class("brick"), BRICK_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("beam"), BEAM_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("shell"), SHELL_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("cseg"), CSEG_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("node"), NODE_DERIVED )

    def test_classes_of_derived_variable(self):
        DISPX_CLASSES = ["node"]
        EFF_STRESS_CLASSES = ["beam", "brick", "shell"]
        VOL_STRAIN_CLASSES = ["brick", "shell"]
        VOLUME_CLASSES = ["brick"]
        self.assertEqual( self.mili.classes_of_derived_variable("disp_x"), DISPX_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("eff_stress"), EFF_STRESS_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("vol_strain"), VOL_STRAIN_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("element_volume"), VOLUME_CLASSES)

    def test_query_multiple_variables(self):
        tolerance = 1e-5
        result = self.mili.query(["pressure", "eff_stress"], "brick", states=[2,44,86], labels=[10,20])

        self.assertEqual(result['pressure']['source'], 'derived')
        self.assertEqual(result['pressure']['class_name'], 'brick')
        self.assertEqual(result['pressure']['title'], 'Pressure')
        self.assertEqual(result['pressure']['layout']['components'], ['pressure'])
        np.testing.assert_equal(result['pressure']['layout']['labels'], [10,20])
        np.testing.assert_allclose(result['pressure']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result['pressure']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][0][0][0], 3.839578e-10, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][0][1][0], -7.582866e-10, delta=tolerance)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][1][0][0], 9.1467084e+02, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][1][1][0], 1.3345321e+03, delta=tolerance)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][2][0][0], 1.01330115e+03, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][2][1][0], 1.9704016e+02, delta=tolerance)

        self.assertEqual(result['eff_stress']['source'], 'derived')
        self.assertEqual(result['eff_stress']['class_name'], 'brick')
        self.assertEqual(result['eff_stress']['title'], 'Effective Stress')
        self.assertEqual(result['eff_stress']['layout']['components'], ['eff_stress'])
        np.testing.assert_equal(result['eff_stress']['layout']['labels'], [10,20])
        np.testing.assert_allclose(result['eff_stress']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result['eff_stress']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][0][0][0], 3.88536481e-10, delta=1e-18)
        self.assertAlmostEqual( result['eff_stress']['data'][0][1][0], 7.60721042e-10, delta=1e-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][1][0][0], 1.88665918e+03, delta=3.5E-7)
        self.assertAlmostEqual( result['eff_stress']['data'][1][1][0], 9.77912305e+03, delta=3.5E-6)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][2][0][0], 1.09058093e+03, delta=3.0E-6)
        self.assertAlmostEqual( result['eff_stress']['data'][2][1][0], 1.22841052e+03, delta=2.5E-6)

    def test_pressure(self):
        """Pressure"""
        tolerance = 1e-5
        result = self.mili.query("pressure", "brick", states=[2,44,86], labels=[10,20])
        self.assertEqual(result['pressure']['source'], 'derived')
        self.assertEqual(result['pressure']['class_name'], 'brick')
        self.assertEqual(result['pressure']['title'], 'Pressure')
        self.assertEqual(result['pressure']['layout']['components'], ['pressure'])
        np.testing.assert_equal(result['pressure']['layout']['labels'], [10,20])
        np.testing.assert_allclose(result['pressure']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result['pressure']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][0][0][0], 3.839578e-10, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][0][1][0], -7.582866e-10, delta=tolerance)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][1][0][0], 9.1467084e+02, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][1][1][0], 1.3345321e+03, delta=tolerance)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['pressure']['data'][2][0][0], 1.01330115e+03, delta=tolerance)
        self.assertAlmostEqual( result['pressure']['data'][2][1][0], 1.9704016e+02, delta=tolerance)

        # Test derived equation for a single integration point
        result = self.mili.query("pressure", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result['pressure']['data'][0][0][0], 1.23563398e+04, delta=5.0E-5)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("pressure", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        np.testing.assert_allclose( result['pressure']['data'][0,:,:], PRESSURE)

    def test_disp_x(self):
        """X Displacement"""
        result = self.mili.query("disp_x", "node", states=[1,50,101], labels=[24,52])
        self.assertEqual(result['disp_x']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_x']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['disp_x']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_x']['data'][1][0][0], 1.00731432e-01)
        self.assertAlmostEqual( result['disp_x']['data'][1][1][0], 0.000000e+00)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_x']['data'][2][0][0], 1.05899870e-01)
        self.assertAlmostEqual( result['disp_x']['data'][2][1][0], 0.000000e+00)

    def test_disp_y(self):
        """Y Displacement"""
        result = self.mili.query("disp_y", "node", states=[1,50,101], labels=[24,52])
        self.assertEqual(result['disp_y']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_y']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['disp_y']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_y']['data'][1][0][0], 5.82261086e-02)
        self.assertAlmostEqual( result['disp_y']['data'][1][1][0], 3.63588333e-06, delta=1e-14)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_y']['data'][2][0][0], 6.11519217e-02)
        self.assertAlmostEqual( result['disp_y']['data'][2][1][0], 2.38418579e-05, delta=1e-13)

    def test_disp_y_reference_state(self):
        """Displacement with specified reference state."""
        result = self.mili.query("disp_y", "node", states=[50], labels=[24,52], reference_state=40)
        self.assertEqual(result['disp_y']['source'], 'derived')
        self.assertAlmostEqual( result['disp_y']['data'][0][0][0], 1.73478723e-02)
        self.assertAlmostEqual( result['disp_y']['data'][0][1][0], -3.99351120e-05, delta=1e-15)

    def test_disp_z(self):
        """Z Displacement"""
        result = self.mili.query("disp_z", "node", states=[1,50,101], labels=[24,52])
        self.assertEqual(result['disp_z']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_z']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['disp_z']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_z']['data'][1][0][0], -6.73872232e-02)
        self.assertAlmostEqual( result['disp_z']['data'][1][1][0], -1.50992393e-01)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_z']['data'][2][0][0], -6.87063932e-02)
        self.assertAlmostEqual( result['disp_z']['data'][2][1][0], -1.55236006e-01)

    def test_disp_mag(self):
        """Displacement Magnitude"""
        result = self.mili.query("disp_mag", "node", states=[1,50,101], labels=[24,52])
        self.assertEqual(result['disp_mag']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['disp_mag']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][1][0][0], 1.34454966e-01)
        self.assertAlmostEqual( result['disp_mag']['data'][1][1][0], 1.50992393e-01)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][2][0][0], 1.40267283e-01)
        self.assertAlmostEqual( result['disp_mag']['data'][2][1][0], 1.55236006e-01)

    def test_disp_mag_ref(self):
        """Displacement Magnitude with reference state"""
        result = self.mili.query("disp_mag", "node", states=[1,50,101], labels=[24,52], reference_state=35)
        self.assertEqual(result['disp_mag']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][0][0][0], 7.08499402e-02)
        self.assertAlmostEqual( result['disp_mag']['data'][0][1][0], 1.03274584e-01)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][1][0][0], 6.91310763e-02)
        self.assertAlmostEqual( result['disp_mag']['data'][1][1][0], 4.77178134e-02)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_mag']['data'][2][0][0], 7.52134994e-02)
        self.assertAlmostEqual( result['disp_mag']['data'][2][1][0], 5.19614369e-02)

    def test_disp_rad_mag_xy(self):
        """Radial (XY) Displacement Magnitude"""
        result = self.mili.query("disp_rad_mag_xy", "node", states=[1,50,101], labels=[24,52])
        self.assertEqual(result['disp_rad_mag_xy']['source'], 'derived')
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][1][0][0], 1.16349049e-01, delta=1e-08)
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][1][1][0], 3.63588333e-06, delta=1e-13)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][2][0][0], 1.22287937e-01, delta=1e-08)
        self.assertAlmostEqual( result['disp_rad_mag_xy']['data'][2][1][0], 2.38418579e-05, delta=1e-12)

    def test_vel_x(self):
        """X Velocity"""
        file_name = os.path.join(dir_path,'data','serial','solids014','solids014_dblplt')
        db = open_database( file_name, suppress_parallel=True )

        # Special case: Just querying first state
        result = db.query("vel_x", "node", states=[1], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_x']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_x']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_x']['data'][0][2][0], 0.000000e+00)

        result = db.query("vel_x", "node", states=[1,2,3], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_x']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_x']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_x']['data'][0][2][0], 0.000000e+00)
        # State 2, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_x']['data'][1][0][0], 1.85962731E-03)
        self.assertAlmostEqual( result['vel_x']['data'][1][1][0], -1.85962731E-03)
        self.assertAlmostEqual( result['vel_x']['data'][1][2][0], 4.429632022E-04)
        # State 3, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_x']['data'][2][0][0], 1.85967549E-03, delta=3.0e-06)
        self.assertAlmostEqual( result['vel_x']['data'][2][1][0], -1.85967549E-03, delta=3.0e-06)
        self.assertAlmostEqual( result['vel_x']['data'][2][2][0], 4.434895803E-04)

    def test_vel_y(self):
        """Y Velocity"""
        file_name = os.path.join(dir_path,'data','serial','solids014','solids014_dblplt')
        db = open_database( file_name, suppress_parallel=True )

        # Special case: Just querying first state
        result = db.query("vel_y", "node", states=[1], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_y']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_y']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_y']['data'][0][2][0], 0.000000e+00)

        result = db.query("vel_y", "node", states=[1,2,3], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_y']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_y']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_y']['data'][0][2][0], 0.000000e+00)
        # State 2, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_y']['data'][1][0][0], 5.83404116E-03)
        self.assertAlmostEqual( result['vel_y']['data'][1][1][0], 5.83404116E-03)
        self.assertAlmostEqual( result['vel_y']['data'][1][2][0], 5.842519924E-02)
        # State 3, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_y']['data'][2][0][0], 5.83404116E-03, delta=2.0E-05)
        self.assertAlmostEqual( result['vel_y']['data'][2][1][0], 5.83404116E-03, delta=2.0E-05)
        self.assertAlmostEqual( result['vel_y']['data'][2][2][0], 5.83521781E-02)

    def test_vel_z(self):
        """Z Velocity"""
        file_name = os.path.join(dir_path,'data','serial','solids014','solids014_dblplt')
        db = open_database( file_name, suppress_parallel=True )

        # Special case: Just querying first state
        result = db.query("vel_z", "node", states=[1], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_z']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_z']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_z']['data'][0][2][0], 0.000000e+00)

        result = db.query("vel_z", "node", states=[1,2,3], labels=[7,1217,3602])
        # State 1, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_z']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_z']['data'][0][1][0], 0.000000e+00)
        self.assertAlmostEqual( result['vel_z']['data'][0][2][0], 0.000000e+00)
        # State 2, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_z']['data'][1][0][0], 8.22748058E-03)
        self.assertAlmostEqual( result['vel_z']['data'][1][1][0], 8.22748058E-03)
        self.assertAlmostEqual( result['vel_z']['data'][1][2][0], -1.0047131E-02)
        # State 3, nodes 7, 1217, 3602
        self.assertAlmostEqual( result['vel_z']['data'][2][0][0], 8.22748058E-03, delta=5.0E-06)
        self.assertAlmostEqual( result['vel_z']['data'][2][1][0], 8.22748058E-03, delta=5.0E-06)
        self.assertAlmostEqual( result['vel_z']['data'][2][2][0], -1.0231257E-02)

    def test_acc_x(self):
        """X Acceleration"""
        # Answers are based on hand calculations, not griz results.
        result = self.mili.query("acc_x", "node", states=[1,50,51,101], labels=[24,52])
        # State 1, nodes 24, 52 (trivial cases, no displacement)
        self.assertAlmostEqual( result['acc_x']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['acc_x']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['acc_x']['data'][1][0][0], -5.22128e+05, delta=7.0)
        self.assertAlmostEqual( result['acc_x']['data'][1][1][0], 0.000000e+00)
        # State 51, nodes 24, 52
        self.assertAlmostEqual( result['acc_x']['data'][2][0][0], -4.953753e+06)
        self.assertAlmostEqual( result['acc_x']['data'][2][1][0], 0.000000e+00)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['acc_x']['data'][3][0][0], 5.405529e+06, delta=4.0)
        self.assertAlmostEqual( result['acc_x']['data'][3][1][0], 0.000000e+00)

        # Griz Comparison
        # These tests only use the forward and backward difference algorithms.
        # Those acceleration calculations in griz may be incorrect causing these tests to fail.
        # file_name = os.path.join(dir_path,'data','serial','solids014','solids014_dblplt')
        # db = open_database( file_name, suppress_parallel=True )
        # result = db.query("acc_x", "node", states=[1,2], labels=[7,1217])
        # # State 1, nodes 7, 1217
        # self.assertAlmostEqual( result['acc_x']['data'][0][0][0], 0.000000e+00)
        # self.assertAlmostEqual( result['acc_x']['data'][0][1][0], 0.000000e+00)
        # # State 3, nodes 7, 1217
        # self.assertAlmostEqual( result['acc_x']['data'][1][0][0], -3.7135E-02, delta=5.0E-16)
        # self.assertAlmostEqual( result['acc_x']['data'][1][1][0], 3.7134E-02, delta=5.0E-16)

    def test_acc_y(self):
        """Y Acceleration"""
        # Answers are based on hand calculations, not griz results.
        result = self.mili.query("acc_y", "node", states=[1,50,51,101], labels=[24,52])
        # State 1, nodes 24, 52 (trivial cases, no displacement)
        self.assertAlmostEqual( result['acc_y']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['acc_y']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result['acc_y']['data'][1][0][0], -3.58829e+05, delta=11.0)
        self.assertAlmostEqual( result['acc_y']['data'][1][1][0], 2.37229e+05, delta=4.0)
        # State 51, nodes 24, 52
        self.assertAlmostEqual( result['acc_y']['data'][2][0][0], -2.8789108e+06, delta=0.05)
        self.assertAlmostEqual( result['acc_y']['data'][2][1][0], -5.6684144e+05, delta=0.003)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result['acc_y']['data'][3][0][0], 3.360507e+06, delta=10.0)
        self.assertAlmostEqual( result['acc_y']['data'][3][1][0], 1.394145e+06, delta=3.0)

    def test_acc_z(self):
        """Z Acceleration"""
        # Answers are based on hand calculations, not griz results.
        result = self.mili.query("acc_z", "node", states=[1,50,51,101], labels=[24,78])
        # State 1, nodes 24, 78 (trivial cases, constant velocity)
        self.assertAlmostEqual( result['acc_z']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result['acc_z']['data'][0][1][0], 0.000000e+00)
        # State 50, nodes 24, 78
        self.assertAlmostEqual( result['acc_z']['data'][1][0][0], 1.995543e+06, delta=14.0)
        self.assertAlmostEqual( result['acc_z']['data'][1][1][0], 2.408091e+06, delta=75.0)
        # State 51, nodes 24, 78
        self.assertAlmostEqual( result['acc_z']['data'][2][0][0], -1.6033686e+05, delta=0.0007)
        self.assertAlmostEqual( result['acc_z']['data'][2][1][0], 1.7476121e+06, delta=0.03)
        # State 101, nodes 24, 78
        self.assertAlmostEqual( result['acc_z']['data'][3][0][0], -1.106856e+06, delta=2.0)
        self.assertAlmostEqual( result['acc_z']['data'][3][1][0], 1.025e+05, delta=20.0)

    def test_vol_strain(self):
        """Strain Trace"""
        result = self.mili.query("vol_strain", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['vol_strain']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['vol_strain']['data'][0][0][0], -1.53583100e-18, delta=5.0E-28)
        self.assertAlmostEqual( result['vol_strain']['data'][0][1][0], 3.03314620e-18, delta=5.0E-27)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['vol_strain']['data'][1][0][0], -3.65868345e-06, delta=4.0E-15)
        self.assertAlmostEqual( result['vol_strain']['data'][1][1][0], -5.33812863e-06, delta=7.0E-16)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['vol_strain']['data'][2][0][0], -4.05320452e-06, delta=5.0E-16)
        self.assertAlmostEqual( result['vol_strain']['data'][2][1][0], -7.88160662e-07, delta=7.0E-17)

        # Test derived equation for a single integration point
        result = self.mili.query("vol_strain", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['vol_strain']['source'], 'derived')
        self.assertAlmostEqual( result['vol_strain']['data'][0][0][0], -4.16194525e-05, delta=1e-13)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("vol_strain", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['vol_strain']['source'], 'derived')
        VOL_STRAIN = np.array([ [-3.15418220e-05, -4.16194525e-05] ])
        np.testing.assert_allclose( result['vol_strain']['data'][0,:,:], VOL_STRAIN)

    def test_prin_strain1(self):
        """1st Principal Strain"""
        result = self.mili.query("prin_strain1", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        # NOTE: For state 2 the answers look different when visually compared because the numbers are very small
        #       which causes the differences in floating point math between Griz (C) and python to be more pronounced.
        self.assertAlmostEqual( result['prin_strain1']['data'][0][0][0], -5.11943684e-19, delta=7.0E-19)
        self.assertAlmostEqual( result['prin_strain1']['data'][0][1][0], 1.01104870e-18, delta=3.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_strain1']['data'][1][0][0], 2.77111781e-06, delta=3.0E-13)
        self.assertAlmostEqual( result['prin_strain1']['data'][1][1][0], 2.17467095e-05, delta=4.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_strain1']['data'][2][0][0], 4.38252954e-07, delta=3.0E-14)
        self.assertAlmostEqual( result['prin_strain1']['data'][2][1][0], 2.66618940e-06, delta=3.0E-15)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain1", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_strain1']['source'], 'derived')
        self.assertAlmostEqual( result['prin_strain1']['data'][0][0][0], 1.52852561e-04, delta=2E-11)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain1", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_strain1']['source'], 'derived')
        PRIN_STRAIN1 = np.array([ [9.27323708e-05, 1.52852561e-04] ])
        np.testing.assert_allclose( result['prin_strain1']['data'][0,:,:], PRIN_STRAIN1, atol=1E-20)

    def test_prin_strain2(self):
        """2nd Principal Strain """
        result = self.mili.query("prin_strain2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        # NOTE: For state 2 the answers look different when visually compared because the numbers are very small
        #       which causes the differences in floating point math between Griz (C) and python to be more pronounced.
        self.assertAlmostEqual( result['prin_strain2']['data'][0][0][0], -5.11943684e-19, delta=2.0E-18)
        self.assertAlmostEqual( result['prin_strain2']['data'][0][1][0], 1.01104870e-18, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_strain2']['data'][1][0][0], -2.66543526e-13, delta=4.0E-13)
        self.assertAlmostEqual( result['prin_strain2']['data'][1][1][0], 2.81440538e-09, delta=4.0E-13)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_strain2']['data'][2][0][0], 1.45898105e-12, delta=9.0E-14)
        self.assertAlmostEqual( result['prin_strain2']['data'][2][1][0], 8.13236145e-09, delta=7.0E-14)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain2", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_strain2']['source'], 'derived')
        self.assertAlmostEqual( result['prin_strain2']['data'][0][0][0], -5.42686757e-06, delta=7E-12)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain2", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_strain2']['source'], 'derived')
        PRIN_STRAIN2 = np.array([ [4.65712728e-05, -5.42686757e-06] ])
        np.testing.assert_allclose( result['prin_strain2']['data'][0,:,:], PRIN_STRAIN2, atol=7E-12)

    def test_prin_strain3(self):
        """3rd Principal Strain """
        result = self.mili.query("prin_strain3", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        # NOTE: For state 2 the answers look different when visually compared because the numbers are very small
        #       which causes the differences in floating point math between Griz (C) and python to be more pronounced.
        self.assertAlmostEqual( result['prin_strain3']['data'][0][0][0], -5.11943684e-19, delta=2.0E-18)
        self.assertAlmostEqual( result['prin_strain3']['data'][0][1][0], 1.01104870e-18, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_strain3']['data'][1][0][0], -6.42980149e-06, delta=3.0E-15)
        self.assertAlmostEqual( result['prin_strain3']['data'][1][1][0], -2.70876499e-05, delta=2.0E-12)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_strain3']['data'][2][0][0], -4.49145864e-06, delta=4.0E-16)
        self.assertAlmostEqual( result['prin_strain3']['data'][2][1][0], -3.46248248e-06, delta=3.0E-13)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain3", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_strain3']['source'], 'derived')
        self.assertAlmostEqual( result['prin_strain3']['data'][0][0][0], -1.89045168e-04, delta=4E-13)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain3", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_strain3']['source'], 'derived')
        PRIN_STRAIN3 = np.array([ [-1.70845480e-04, -1.89045168e-04] ])
        np.testing.assert_allclose( result['prin_strain3']['data'][0,:,:], PRIN_STRAIN3, atol=4E-13)

    def test_prin_dev_strain1(self):
        """1st Principal Deviatoric Strain """
        result = self.mili.query("prin_dev_strain1", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][0][0][0], 0.000000000e+00, delta=7.0E-19)
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][0][1][0], 0.000000000e+00, delta=3.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][1][0][0], 3.99067903e-06, delta=4.0E-15)
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][1][1][0], 2.35260850e-05, delta=2.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][2][0][0], 1.78932112e-06, delta=2.0E-13)
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][2][1][0], 2.92890968e-06, delta=2.0E-16)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_strain1", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_strain1']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_strain1']['data'][0][0][0], 1.66725717e-04, delta=2E-11)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_strain1", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_dev_strain1']['source'], 'derived')
        PRIN_DEV_STRAIN1 = np.array([ [1.03246311e-04, 1.66725717e-04] ])
        np.testing.assert_allclose( result['prin_dev_strain1']['data'][0,:,:], PRIN_DEV_STRAIN1, atol=2E-11)

    def test_prin_dev_strain2(self):
        """2nd Principal Deviatoric Strain """
        result = self.mili.query("prin_dev_strain2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][0][0][0], 0.000000000e+00, delta=6.0E-19)
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][0][1][0], 0.000000000e+00, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][1][0][0], 1.21956089e-06, delta=4.0E-13)
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][1][1][0], 1.78219057e-06, delta=4.0E-13)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][2][0][0], 1.35106961e-06, delta=4.0E-15)
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][2][1][0], 2.70852581e-07, delta=6.0E-14)

        # Test derived equation for a single integration point - FAILS
        result = self.mili.query("prin_dev_strain2", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_strain2']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_strain2']['data'][0][0][0], 8.44628357e-06, delta=8.0E-12)

        # Test derived equation when there are multiple integration points - FAILS
        result = self.mili.query("prin_dev_strain2", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_dev_strain2']['source'], 'derived')
        PRIN_DEV_STRAIN2 = np.array([ [5.70852135e-05, 8.44628357e-06] ])
        np.testing.assert_allclose( result['prin_dev_strain2']['data'][0,:,:], PRIN_DEV_STRAIN2, atol=8.0E-12)

    def test_prin_dev_strain3(self):
        """3rd Principal Deviatoric Strain """
        result = self.mili.query("prin_dev_strain3", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][0][0][0], 0.000000000e+00, delta=2.0E-18)
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][0][1][0], 0.000000000e+00, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][1][0][0], -5.21024049e-06, delta=5.0E-13)
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][1][1][0], -2.53082744e-05, delta=2.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][2][0][0], -3.14039062e-06, delta=3.0E-15)
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][2][1][0], -3.19976220e-06, delta=5.0E-15)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_strain3", "shell", labels=[4], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_strain3']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_strain3']['data'][0][0][0], -1.75172012e-04, delta=5E-14)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_strain3", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_dev_strain3']['source'], 'derived')
        PRIN_DEV_STRAIN3 = np.array([ [-1.60331532e-4, -1.75172012e-04] ])
        np.testing.assert_allclose( result['prin_dev_strain3']['data'][0,:,:], PRIN_DEV_STRAIN3, atol=5E-14)

    def test_prin_stress1(self):
        """1st Principal Stress"""
        result = self.mili.query("prin_stress1", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_stress1']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_stress1']['data'][0][0][0], -2.43397912e-10, delta=1e-16)
        self.assertAlmostEqual( result['prin_stress1']['data'][0][1][0], 1.26490828e-09, delta=1e-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_stress1']['data'][1][0][0], 6.25506591e+00, delta=2.0E-5)
        self.assertAlmostEqual( result['prin_stress1']['data'][1][1][0], 4.09456445e+03, delta=4.0E-6)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_stress1']['data'][2][0][0], -6.00380859e+02, delta=4.0E-7)
        self.assertAlmostEqual( result['prin_stress1']['data'][2][1][0], 4.78862060e+02, delta=1.0E-6)

        # Test derived equation when there are multiple integration points - Shell Element
        result = self.mili.query("prin_stress1", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_stress1']['source'], 'derived')
        PRIN_STRESS1 = np.array([ [1.59406152e+03, 2.80703052e+03] ])
        np.testing.assert_allclose( result['prin_stress1']['data'][0,:,:], PRIN_STRESS1, atol=3E-4 )

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress1", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['prin_stress1']['source'], 'derived')
        self.assertAlmostEqual( result['prin_stress1']['data'][0][0][0], 1.65892188e+03, delta=5.0E-3)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress1", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['prin_stress1']['source'], 'derived')
        PRIN_STRESS1 = np.array([ [1.23608057e+03, 1.65892188e+03, 1.73505859e+03, 1.32386670e+03] ])
        np.testing.assert_allclose( result['prin_stress1']['data'][0,:,:], PRIN_STRESS1, rtol=4.0E-06)

    def test_prin_stress2(self):
        """2nd Principal Stress"""
        result = self.mili.query("prin_stress2", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_stress2']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_stress2']['data'][0][0][0], -2.65816896e-10)
        self.assertAlmostEqual( result['prin_stress2']['data'][0][1][0], 5.24967625e-10)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_stress2']['data'][1][0][0], -6.33233765e+02, delta=2.0e-4)
        self.assertAlmostEqual( result['prin_stress2']['data'][1][1][0], -9.23257324e+02, delta=1.5E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_stress2']['data'][2][0][0], -7.01515808e+02, delta=1.0E-4)
        self.assertAlmostEqual( result['prin_stress2']['data'][2][1][0], -1.34535721e+02, delta=2.0E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress2", "shell", labels=[2,11], states=[44], ips=[1])
        self.assertEqual(result['prin_stress2']['source'], 'derived')
        self.assertAlmostEqual( result['prin_stress2']['data'][0][0][0], 1.24129639e+03, delta=4.0E-06)
        self.assertAlmostEqual( result['prin_stress2']['data'][0][1][0], -8.73693164e+03, delta=1.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress2", "shell", labels=[2], states=[44], ips=[1,2])
        self.assertEqual(result['prin_stress2']['source'], 'derived')
        PRIN_STRESS2 = np.array([ [1.24129639e+03, -2.53508789e+03] ])
        np.testing.assert_allclose( result['prin_stress2']['data'][0,:,:], PRIN_STRESS2)

    def test_prin_stress3(self):
        """3rd Principal Stress"""
        result = self.mili.query("prin_stress3", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_stress3']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_stress3']['data'][0][0][0], -6.42658482e-10, delta=1e-19)
        self.assertAlmostEqual( result['prin_stress3']['data'][0][1][0], 4.84983775e-10, delta=1e-16)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_stress3']['data'][1][0][0], -2.11703394e+03, delta=5.0E-6)
        self.assertAlmostEqual( result['prin_stress3']['data'][1][1][0], -7.17490381e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_stress3']['data'][2][0][0], -1.73800671e+03, delta=2.0E-4)
        self.assertAlmostEqual( result['prin_stress3']['data'][2][1][0], -9.35446838e+02, delta=1.0E-4)

        # Test derived equation when there are multiple integration points - Shell Element
        result = self.mili.query("prin_stress3", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['prin_stress3']['source'], 'derived')
        PRIN_STRESS3 = np.array([ [-4.48850391e+03, -5.08291699e+03] ])
        np.testing.assert_allclose( result['prin_stress3']['data'][0,:,:], PRIN_STRESS3, rtol=1.0E-07)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress3", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['prin_stress3']['source'], 'derived')
        self.assertAlmostEqual( result['prin_stress3']['data'][0][0][0], -3.87279453e+04, delta=2.0E-5)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress3", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['prin_stress3']['source'], 'derived')
        PRIN_STRESS3 = np.array([ [-2.40547637e+04, -3.87279453e+04, -3.81615352e+04, -2.34779512e+04] ])
        np.testing.assert_allclose( result['prin_stress3']['data'][0,:,:], PRIN_STRESS3, rtol=2.0E-07)

    def test_eff_stress(self):
        """Effective (von Mises) Stress"""
        result = self.mili.query("eff_stress", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['eff_stress']['source'], 'derived')
        self.assertEqual(result['eff_stress']['class_name'], 'brick')
        self.assertEqual(result['eff_stress']['title'], 'Effective Stress')
        self.assertEqual(result['eff_stress']['layout']['components'], ['eff_stress'])
        np.testing.assert_equal(result['eff_stress']['layout']['labels'], [10,20])
        np.testing.assert_allclose(result['eff_stress']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result['eff_stress']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][0][0][0], 3.88536481e-10, delta=1e-18)
        self.assertAlmostEqual( result['eff_stress']['data'][0][1][0], 7.60721042e-10, delta=1e-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][1][0][0], 1.88665918e+03, delta=3.5E-7)
        self.assertAlmostEqual( result['eff_stress']['data'][1][1][0], 9.77912305e+03, delta=3.5E-6)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['eff_stress']['data'][2][0][0], 1.09058093e+03, delta=3.0E-6)
        self.assertAlmostEqual( result['eff_stress']['data'][2][1][0], 1.22841052e+03, delta=2.5E-6)

        # Test derived equation when there are multiple integration points - Shell Element
        result = self.mili.query("eff_stress", "shell", labels=[4], states=[44], ips=[1,2])
        self.assertEqual(result['eff_stress']['source'], 'derived')
        EFF_STRESS = np.array([ [5.62608936e+03, 6.83914795e+03] ])
        np.testing.assert_allclose( result['eff_stress']['data'][0,:,:], EFF_STRESS, rtol=1.0E-07)

        # Test derived equation for a single integration point
        result = self.mili.query("eff_stress", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['eff_stress']['source'], 'derived')
        self.assertAlmostEqual( result['eff_stress']['data'][0][0][0], 3.95834883e+04, delta=2.0E-05)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("eff_stress", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['eff_stress']['source'], 'derived')
        EFF_STRESS = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        np.testing.assert_allclose( result['eff_stress']['data'][0,:,:], EFF_STRESS)

    def test_prin_dev_stress1(self):
        """1st Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress1", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_dev_stress1']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][0][0][0], 1.40559855e-10, delta=2.0E-17)
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][0][1][0], 5.06621689e-10, delta=6.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][1][0][0], 9.20925903e+02, delta=3.5E-7)
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][1][1][0], 5.42909668e+03, delta=3.5E-7)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][2][0][0], 4.12920258e+02, delta=3.5E-5)
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][2][1][0], 6.75902222e+02, delta=3.5E-7)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress1", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_stress1']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_stress1']['data'][0][0][0], 1.40152617e+04, delta=5.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress1", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['prin_dev_stress1']['source'], 'derived')
        PRIN_DEV_STRESS1 = np.array([ [8.84230957e+03, 1.40152617e+04, 1.38772188e+04, 8.70856152e+03] ])
        np.testing.assert_allclose( result['prin_dev_stress1']['data'][0,:,:], PRIN_DEV_STRESS1, rtol=5.0E-07)

    def test_prin_dev_stress2(self):
        """2nd Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress2", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_dev_stress2']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][0][0][0], 1.18140858e-10, delta=2.5E-19)
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][0][1][0], -2.33318975e-10, delta=6.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][1][0][0], 2.81437103e+02, delta=6.5E-5)
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][1][1][0], 4.11274780e+02, delta=1.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][2][0][0], 3.11785278e+02, delta=3.5E-5)
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][2][1][0], 6.25044403e+01, delta=1.5E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress2", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_stress2']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_stress2']['data'][0][0][0], 1.23563418e+04, delta=2.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress2", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['prin_dev_stress2']['source'], 'derived')
        PRIN_DEV_STRESS2 = np.array([ [7.60622510e+03, 1.23563418e+04, 1.21421543e+04, 7.38469531e+03] ])
        np.testing.assert_allclose( result['prin_dev_stress2']['data'][0,:,:], PRIN_DEV_STRESS2, rtol=6.0E-07)

    def test_prin_dev_stress3(self):
        """3rd Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress3", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['prin_dev_stress3']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][0][0][0], -2.58700728e-10, delta=5.0E-19)
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][0][1][0], -2.73302797e-10, delta=3.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][1][0][0], -1.20236304e+03, delta=1.5E-4)
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][1][1][0], -5.84037158e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][2][0][0], -7.24705566e+02, delta=1.5E-4)
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][2][1][0], -7.38406677e+02, delta=6.5E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress3", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['prin_dev_stress3']['source'], 'derived')
        self.assertAlmostEqual( result['prin_dev_stress3']['data'][0][0][0], -2.63716055e+04, delta=4.0E-05)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress3", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['prin_dev_stress3']['source'], 'derived')
        PRIN_DEV_STRESS3 = np.array([ [-1.64485352e+04, -2.63716055e+04, -2.60193750e+04, -1.60932568e+04] ])
        np.testing.assert_allclose( result['prin_dev_stress3']['data'][0,:,:], PRIN_DEV_STRESS3, rtol=2.0E-07)

    def test_max_shear_stress(self):
        """Max. Shear Stress"""
        result = self.mili.query("max_shear_stress", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['max_shear_stress']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['max_shear_stress']['data'][0][0][0], 1.99630285e-10, delta=1.5E-17)
        self.assertAlmostEqual( result['max_shear_stress']['data'][0][1][0], 3.89962229e-10, delta=2.0E-19)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['max_shear_stress']['data'][1][0][0], 1.06164453e+03, delta=1.5E-6)
        self.assertAlmostEqual( result['max_shear_stress']['data'][1][1][0], 5.63473438e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['max_shear_stress']['data'][2][0][0], 5.68812927e+02, delta=6.5E-5)
        self.assertAlmostEqual( result['max_shear_stress']['data'][2][1][0], 7.07154419e+02, delta=5.5E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("max_shear_stress", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['max_shear_stress']['source'], 'derived')
        self.assertAlmostEqual( result['max_shear_stress']['data'][0][0][0], 2.01934336e+04, delta=2.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("max_shear_stress", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['max_shear_stress']['source'], 'derived')
        MAX_SHEAR_STRESS = np.array([ [1.26454219e+04, 2.01934336e+04, 1.99482969e+04, 1.24009092e+04] ])
        np.testing.assert_allclose( result['max_shear_stress']['data'][0,:,:], MAX_SHEAR_STRESS, rtol=2.0E-07)

    def test_triaxiality(self):
        """Triaxiality"""
        # Answers are based on hand calculations using griz results for pressure and seff.
        result = self.mili.query("triaxiality", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['triaxiality']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['triaxiality']['data'][0][0][0], -3.83957754/3.88536481, delta=1.0E-08)
        self.assertAlmostEqual( result['triaxiality']['data'][0][1][0], +7.58286600/7.60721042, delta=1.0E-06)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['triaxiality']['data'][1][0][0], -9.14670837e+02/1.88665918e+03, delta=1.0E-8)
        self.assertAlmostEqual( result['triaxiality']['data'][1][1][0], -1.33453210e+03/9.77912305e+03, delta=1.0E-8)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['triaxiality']['data'][2][0][0], -1.01330109e+03/1.09058093e+03, delta=7.0E-8)
        self.assertAlmostEqual( result['triaxiality']['data'][2][1][0], -1.97040161e+02/1.22841052e+03, delta=5.5E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("triaxiality", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['triaxiality']['source'], 'derived')
        self.assertAlmostEqual( result['triaxiality']['data'][0][0][0], -1.23563398e+04/3.95834883e+04, delta=1.0E-08)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("triaxiality", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['triaxiality']['source'], 'derived')
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        SEFF = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        TRIAXIALITY = -PRESSURE/SEFF
        np.testing.assert_allclose( result['triaxiality']['data'][0,:,:], TRIAXIALITY, rtol=1.0E-07)

    def test_normalized_pressure(self):
        """Normalized Pressure"""
        # Answers are based on hand calculations using griz results for pressure and seff.
        result = self.mili.query("norm_press", "brick", states=[2,44,86], labels=[10, 20])
        self.assertEqual(result['norm_press']['source'], 'derived')
        # State 2, labels 10, 20
        self.assertAlmostEqual( result['norm_press']['data'][0][0][0], +3.83957754/3.88536481, delta=1.0E-08)
        self.assertAlmostEqual( result['norm_press']['data'][0][1][0], -7.58286600/7.60721042, delta=1.0E-06)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result['norm_press']['data'][1][0][0], +9.14670837e+02/1.88665918e+03, delta=1.0E-8)
        self.assertAlmostEqual( result['norm_press']['data'][1][1][0], +1.33453210e+03/9.77912305e+03, delta=1.0E-8)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result['norm_press']['data'][2][0][0], +1.01330109e+03/1.09058093e+03, delta=7.0E-8)
        self.assertAlmostEqual( result['norm_press']['data'][2][1][0], +1.97040161e+02/1.22841052e+03, delta=5.5E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("norm_press", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result['norm_press']['source'], 'derived')
        self.assertAlmostEqual( result['norm_press']['data'][0][0][0], +1.23563398e+04/3.95834883e+04, delta=1.0E-08)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("norm_press", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        self.assertEqual(result['norm_press']['source'], 'derived')
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        SEFF = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        NORMALIZED_PRESSURE = PRESSURE/SEFF
        np.testing.assert_allclose( result['norm_press']['data'][0,:,:], NORMALIZED_PRESSURE, rtol=1.0E-07)

    def test_eps_rate(self):
        """Effective Plastic Strain Rate"""
        # Use a different database that has plastic strain (eps)
        # This database does not have (i.e. test) variable time steps.
        file_name = os.path.join(dir_path,'data','serial','d3samp4','d3samp4.plt')
        db = open_database( file_name, suppress_parallel=True )

        # Hand calcs for eps_rate
        EPS_RATE1_46_1 = 0.0
        EPS_RATE2_46_1 = 0.5 * ( (1.95459e-2 - 0.0)/1000 + (5.01047e-2 - 1.95459e-2)/1000 )
        EPS_RATE3_46_1 = 0.5 * ( (5.01047e-2 - 1.95459e-2)/1000 + (8.30480e-2 - 5.01047e-2)/1000 )
        EPS_RATE11_46_1 = (8.3048e-2 - 8.3048e-2)/1000  # trivial test of the last state, eps_rate=0

        # Test for a single integration point
        result = db.query("eps_rate", "shell", states=[3], labels=[46], ips=[1])
        self.assertAlmostEqual( result['eps_rate']['data'][0][0][0], EPS_RATE3_46_1, delta=1E-10)

        # Test for multiple states (includes first and last states)
        result = db.query("eps_rate", "shell", states=[1, 2, 3, 11], labels=[46], ips=[1])
        EPS_RATE_STATES = np.array( [EPS_RATE1_46_1, EPS_RATE2_46_1, EPS_RATE3_46_1, EPS_RATE11_46_1] )
        np.testing.assert_allclose( result['eps_rate']['data'][:,0,0], EPS_RATE_STATES, rtol=2.0E-06)

        # Test for multiple integration points
        result = db.query("eps_rate", "shell", states=[2], labels=[46], ips=[1,2])
        EPS_RATE2_46_2 = 0.5 * ( (1.39783e-3 - 0.0)/1000 + (2.11001e-3 - 1.39783e-3)/1000 )
        EPS_RATE_STATES = np.array( [EPS_RATE2_46_1, EPS_RATE2_46_2] )
        np.testing.assert_allclose( result['eps_rate']['data'][0,0,:], EPS_RATE_STATES, rtol=1.0E-06)

        # Test for multiple elements
        result = db.query("eps_rate", "shell", states=[2], labels=[46,47], ips=[1])
        EPS_RATE2_47_1 = 0.5 * ( (1.32413e-2 - 0.0)/1000 + (3.40781e-2 - 1.32413e-2)/1000 )
        EPS_RATE_STATES = np.array( [EPS_RATE2_46_1, EPS_RATE2_47_1] )
        np.testing.assert_allclose( result['eps_rate']['data'][0,:,0], EPS_RATE_STATES, rtol=1.0E-06)

    def test_nodetangmag(self):
        """Nodal Tangential Traction Magnitude"""
        # Uses a different database that contains the primals "nodtang"
        file_name = os.path.join(dir_path,'data','serial','dbl_nodtang','dblplt')
        db = open_database( file_name, suppress_parallel=True )

        result = db.query("nodtangmag", "cbs1_particle", labels=[5, 95, 115], states=[1, 60, 122])
        # cbs1_particle 5
        self.assertAlmostEqual(result['nodtangmag']['data'][0,0,0], 7.362608292957173e-14, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,0,0], 2.091464039505183, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,0,0], 2.091464039505183, delta=1.0e-20 )
        # cbs1_particle 95
        self.assertAlmostEqual(result['nodtangmag']['data'][0,1,0], 3.39893903983239e-13, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,1,0], 57.39554750267382, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,1,0], 72.4122659097881, delta=1.0e-20 )
        # cbs1_particle 115
        self.assertAlmostEqual(result['nodtangmag']['data'][0,2,0], 4.716543468420842e-13, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,2,0], 36.95306253524787, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,2,0], 36.41065450783031, delta=1.0e-20 )

        result = db.query("nodtangmag", "cbs1_quad", labels=[51, 57, 60], states=[1, 60, 122])
        # cbs1_quad 51
        self.assertAlmostEqual(result['nodtangmag']['data'][0,0,0], 1.385230111634118e-14, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,0,0], 70.13199989140008, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,0,0], 78.51272797578739, delta=1.0e-20 )
        # cbs1_quad 57
        self.assertAlmostEqual(result['nodtangmag']['data'][0,1,0], 7.988411450852106e-13, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,1,0], 10.36301416996304, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,1,0], 10.36301416996304, delta=1.0e-20 )
        # cbs1_quad 60
        self.assertAlmostEqual(result['nodtangmag']['data'][0,2,0], 1.3815008478712372e-12, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][1,2,0], 62.646706196877986, delta=1.0e-20 )
        self.assertAlmostEqual(result['nodtangmag']['data'][2,2,0], 75.71470466999355, delta=1.0e-20 )

    def test_mat_cog_disp(self):
        """Test Material Center of Gravity Displacement."""
        # Uses a different database that contains the primals "matcg[x|y|z]"
        file_name = os.path.join(dir_path,'data','serial','rigid_body_1','rigid_body1.plt')
        db = open_database( file_name, suppress_parallel=True )

        result = db.query("mat_cog_disp_x", "mat", labels=[1], states=[1,3,5,7])
        self.assertAlmostEqual(result['mat_cog_disp_x']['data'][0,0,0], 0.0)
        self.assertAlmostEqual(result['mat_cog_disp_x']['data'][1,0,0], 0.0)
        self.assertAlmostEqual(result['mat_cog_disp_x']['data'][2,0,0], 0.0)
        self.assertAlmostEqual(result['mat_cog_disp_x']['data'][3,0,0], 0.0)

        result = db.query("mat_cog_disp_y", "mat", labels=[1], states=[1,3,5,7])
        self.assertAlmostEqual(result['mat_cog_disp_y']['data'][0,0,0], 0.0)
        self.assertAlmostEqual(result['mat_cog_disp_y']['data'][1,0,0], 0.1004999, delta=1.0e-7)
        self.assertAlmostEqual(result['mat_cog_disp_y']['data'][2,0,0], 0.1999999, delta=1.0e-7)
        self.assertAlmostEqual(result['mat_cog_disp_y']['data'][3,0,0], 0.3, delta=1.0e-7)

        result = db.query("mat_cog_disp_z", "mat", labels=[1], states=[1,3,5,7])
        self.assertAlmostEqual(result['mat_cog_disp_z']['data'][0,0,0], 0.0)
        self.assertAlmostEqual(result['mat_cog_disp_z']['data'][1,0,0], 0.1004999, delta=1.0e-7)
        self.assertAlmostEqual(result['mat_cog_disp_z']['data'][2,0,0], 0.1999999, delta=1.0e-7)
        self.assertAlmostEqual(result['mat_cog_disp_z']['data'][3,0,0], 0.3, delta=1.0e-7)

    def test_hex_element_volume(self):
        """Test Element Volume calculation for Hexes."""
        result = self.mili.query("element_volume", "brick", labels=[1,4], states=[1,3,4])

        np.testing.assert_equal( result["element_volume"]["layout"]["labels"], [1,4])
        np.testing.assert_equal( result["element_volume"]["layout"]["states"], [1,3,4])
        np.testing.assert_allclose( result["element_volume"]["layout"]["times"], [0.0e00, 2.0e-05, 3.0e-05])
        self.assertEqual( result["element_volume"]["source"], "derived")
        self.assertEqual( result["element_volume"]["class_name"], "brick")
        self.assertEqual( result["element_volume"]["title"], "Element Volume")

        self.assertAlmostEqual(result["element_volume"]["data"][0,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][0,1,0], 0.0291667)
        self.assertAlmostEqual(result["element_volume"]["data"][1,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][1,1,0], 0.0291667)
        self.assertAlmostEqual(result["element_volume"]["data"][2,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][2,1,0], 0.0291667)

    def test_tet_element_volume(self):
        """Test Element Volume calculation for Tets."""
        # Uses a different database that contains the tet elements"
        file_name = os.path.join(dir_path,'data','serial','tet','tet1_t4.plt')
        db = open_database( file_name, suppress_parallel=True )

        result = db.query("element_volume", "tet", labels=[53,59,65], states=[1,49,81])

        np.testing.assert_equal( result["element_volume"]["layout"]["labels"], [53,59,65])
        np.testing.assert_equal( result["element_volume"]["layout"]["states"], [1,49,81])
        np.testing.assert_allclose( result["element_volume"]["layout"]["times"], [0.0,  48.0, 80.0])
        self.assertEqual( result["element_volume"]["source"], "derived")
        self.assertEqual( result["element_volume"]["class_name"], "tet")
        self.assertEqual( result["element_volume"]["title"], "Element Volume")

        self.assertAlmostEqual(result["element_volume"]["data"][0,0,0], 6.291125304843772E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][0,1,0], 4.703088313429738E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][0,2,0], 4.166676663023120E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][1,0,0], 6.208914054162492E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][1,1,0], 4.682676728095200E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][1,2,0], 4.158493911287766E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][2,0,0], 6.239515600934483E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][2,1,0], 4.717303015668719E-05)
        self.assertAlmostEqual(result["element_volume"]["data"][2,2,0], 4.147314159979739E-05)

    def test_quad_area(self):
        """Test area calculation for quad elements."""
        result = self.mili.query("area", "cseg", labels=[1,12,24], states=[1,40,80])

        np.testing.assert_equal( result["area"]["layout"]["labels"], [1,12,24])
        np.testing.assert_equal( result["area"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["area"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["area"]["source"], "derived")
        self.assertEqual( result["area"]["class_name"], "cseg")
        self.assertEqual( result["area"]["title"], "Quad Area")

        # State 1
        self.assertAlmostEqual(result["area"]["data"][0,0,0], 0.078125)
        self.assertAlmostEqual(result["area"]["data"][0,1,0], 0.171875)
        self.assertAlmostEqual(result["area"]["data"][0,2,0], 0.17187496)
        # State 40
        self.assertAlmostEqual(result["area"]["data"][1,0,0], 0.07812971)
        self.assertAlmostEqual(result["area"]["data"][1,1,0], 0.17187835)
        self.assertAlmostEqual(result["area"]["data"][1,2,0], 0.17187496)
        # State 80
        self.assertAlmostEqual(result["area"]["data"][2,0,0], 0.07812706)
        self.assertAlmostEqual(result["area"]["data"][2,1,0], 0.17187445)
        self.assertAlmostEqual(result["area"]["data"][2,2,0], 0.17187496)

    def test_hex_centroid(self):
        """Test centroid calculation for Hex elements."""
        result = self.mili.query("centroid", "brick", labels=[1,12], states=[1,40,80])

        np.testing.assert_equal( result["centroid"]["layout"]["labels"], [1,12])
        np.testing.assert_equal( result["centroid"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["centroid"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["centroid"]["source"], "derived")
        self.assertEqual( result["centroid"]["class_name"], "brick")
        self.assertEqual( result["centroid"]["title"], "Centroid Position")

        # State 1
        np.testing.assert_allclose(result["centroid"]["data"][0,0,:], [0.583133  , 0.15625003, 2.3333337 ])
        np.testing.assert_allclose(result["centroid"]["data"][0,1,:], [0.21874999, 0.81638616, 2.6000001 ])
        # State 40
        np.testing.assert_allclose(result["centroid"]["data"][1,0,:], [0.583133  , 0.15625003, 2.0106943 ])
        np.testing.assert_allclose(result["centroid"]["data"][1,1,:], [0.21874999, 0.81638616, 2.2773683 ])
        # State 80
        np.testing.assert_allclose(result["centroid"]["data"][2,0,:], [0.583133  , 0.15625003, 1.9905574 ])
        np.testing.assert_allclose(result["centroid"]["data"][2,1,:], [0.21874999, 0.81638616, 2.2572217 ])

    def test_beam_centroid(self):
        """Test centroid calculation for Beam elements."""
        result = self.mili.query("centroid", "beam", labels=[1,12], states=[1,40,80])

        np.testing.assert_equal( result["centroid"]["layout"]["labels"], [1,12])
        np.testing.assert_equal( result["centroid"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["centroid"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["centroid"]["source"], "derived")
        self.assertEqual( result["centroid"]["class_name"], "beam")
        self.assertEqual( result["centroid"]["title"], "Centroid Position")

        # State 1
        np.testing.assert_allclose(result["centroid"]["data"][0,0,:], [0.9330127 , 0.25      , 0.        ])
        np.testing.assert_allclose(result["centroid"]["data"][0,1,:], [0.8668914 , 0.50049996, 0.1       ])
        # State 40
        np.testing.assert_allclose(result["centroid"]["data"][1,0,:], [0.9330127 , 0.25      , 0.        ])
        np.testing.assert_allclose(result["centroid"]["data"][1,1,:], [0.86924446, 0.50186634, 0.09236038])
        # State 80
        np.testing.assert_allclose(result["centroid"]["data"][2,0,:], [0.9330127 , 0.25      , 0.        ])
        np.testing.assert_allclose(result["centroid"]["data"][2,1,:], [0.87171483, 0.50328773, 0.08988625])

    def test_shell_centroid(self):
        """Test centroid calculation for Shell elements."""
        result = self.mili.query("centroid", "shell", labels=[1,12], states=[1,40,80])

        np.testing.assert_equal( result["centroid"]["layout"]["labels"], [1,12])
        np.testing.assert_equal( result["centroid"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["centroid"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["centroid"]["source"], "derived")
        self.assertEqual( result["centroid"]["class_name"], "shell")
        self.assertEqual( result["centroid"]["title"], "Centroid Position")

        # State 1
        np.testing.assert_allclose(result["centroid"]["data"][0,0,:], [0.5831329 , 0.15625003, 2.        ])
        np.testing.assert_allclose(result["centroid"]["data"][0,1,:], [0.34374994, 1.2828925 , 2.        ])
        # State 40
        np.testing.assert_allclose(result["centroid"]["data"][1,0,:], [0.5831725 , 0.15626055, 1.8761616 ])
        np.testing.assert_allclose(result["centroid"]["data"][1,1,:], [0.3437595 , 1.2829274 , 1.8770695 ])
        # State 80
        np.testing.assert_allclose(result["centroid"]["data"][2,0,:], [0.5831527 , 0.1562534 , 1.8437902 ])
        np.testing.assert_allclose(result["centroid"]["data"][2,1,:], [0.3437514 , 1.2829016 , 1.8467228 ])

    def test_node_centroid(self):
        """Test centroid calculation for Node elements."""
        result = self.mili.query("centroid", "node", labels=[1,12], states=[1,40,80])

        np.testing.assert_equal( result["centroid"]["layout"]["labels"], [1,12])
        np.testing.assert_equal( result["centroid"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["centroid"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["centroid"]["source"], "derived")
        self.assertEqual( result["centroid"]["class_name"], "node")
        self.assertEqual( result["centroid"]["title"], "Centroid Position")

        # State 1
        np.testing.assert_allclose(result["centroid"]["data"][0,0,:], [1.       , 0.       , 0.       ])
        np.testing.assert_allclose(result["centroid"]["data"][0,1,:], [1.       , 0.       , 2.       ])
        # State 40
        np.testing.assert_allclose(result["centroid"]["data"][1,0,:], [1.       , 0.       , 0.       ])
        np.testing.assert_allclose(result["centroid"]["data"][1,1,:], [1.0000436, 0.       , 1.8774046])
        # State 80
        np.testing.assert_allclose(result["centroid"]["data"][2,0,:], [1.       , 0.       , 0.       ])
        np.testing.assert_allclose(result["centroid"]["data"][2,1,:], [1.000033 , 0.       , 1.8454171])

    def test_surfstrain(self):
        """Test derived result surfstrain."""
        file_name = os.path.join(dir_path,'data','serial','dbl_nodtang','dblplt')
        db = open_database( file_name, suppress_parallel=True )

        with self.assertRaises(MiliPythonError):
            # Missing face number
            result = db.query("surfstrainx", "brick", labels=[71,72,90], states=[21,22,23])
        with self.assertRaises(MiliPythonError):
            # Face number is invalid
            result = db.query("surfstrainx", "brick", labels=[71,72,90], states=[21,22,23], face=7)

        result = db.query("surfstrainx", "brick", labels=[71,72,90], states=[21,22,23], face=1)
        # State 21
        self.assertAlmostEqual(result["surfstrainx"]["data"][0,0,0], 7.01262428e-06)
        self.assertAlmostEqual(result["surfstrainx"]["data"][0,1,0], 1.16926170e-05)
        self.assertAlmostEqual(result["surfstrainx"]["data"][0,2,0], 5.40478068e-06)
        # State 22
        self.assertAlmostEqual(result["surfstrainx"]["data"][1,0,0], 7.61965573e-06)
        self.assertAlmostEqual(result["surfstrainx"]["data"][1,1,0], 1.35642446e-05)
        self.assertAlmostEqual(result["surfstrainx"]["data"][1,2,0], 6.20267215e-06)
        # State 23
        self.assertAlmostEqual(result["surfstrainx"]["data"][2,0,0], 9.98206399e-06)
        self.assertAlmostEqual(result["surfstrainx"]["data"][2,1,0], 1.77060717e-05)
        self.assertAlmostEqual(result["surfstrainx"]["data"][2,2,0], 6.20345624e-06)

        with self.assertRaises(MiliPythonError):
            # Missing face number
            result = db.query("surfstrainy", "brick", labels=[71,72,90], states=[21,22,23])
        with self.assertRaises(MiliPythonError):
            # Face number is invalid
            result = db.query("surfstrainy", "brick", labels=[71,72,90], states=[21,22,23], face=7)

        result = db.query("surfstrainy", "brick", labels=[71,72,90], states=[21,22,23], face=3)
        # State 21
        self.assertAlmostEqual(result["surfstrainy"]["data"][0,0,0], 0.00028667)
        self.assertAlmostEqual(result["surfstrainy"]["data"][0,1,0], 0.00028984)
        self.assertAlmostEqual(result["surfstrainy"]["data"][0,2,0], 0.00014372)
        # State 22
        self.assertAlmostEqual(result["surfstrainy"]["data"][1,0,0], 0.00027829)
        self.assertAlmostEqual(result["surfstrainy"]["data"][1,1,0], 0.00034462)
        self.assertAlmostEqual(result["surfstrainy"]["data"][1,2,0], 0.00024833)
        # State 23
        self.assertAlmostEqual(result["surfstrainy"]["data"][2,0,0], 0.00040497)
        self.assertAlmostEqual(result["surfstrainy"]["data"][2,1,0], 0.00053024)
        self.assertAlmostEqual(result["surfstrainy"]["data"][2,2,0], 0.00036156)


class ParallelDerivedExpressions(unittest.TestCase):
    """Test Parallel implementation of Derived Expressions Wrapper class."""
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = open_database( ParallelDerivedExpressions.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

    def test_pressure(self):
        """Pressure"""
        tolerance = 1e-5
        result = self.mili.query("pressure", "brick", states=[2,44,86], labels=[10,20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['pressure']['data'][0][0][0], 3.83957754e-10, delta=tolerance)
        self.assertAlmostEqual( result[5]['pressure']['data'][0][0][0], -7.5828660e-10, delta=tolerance)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['pressure']['data'][1][0][0], 9.14670837e+02, delta=tolerance)
        self.assertAlmostEqual( result[5]['pressure']['data'][1][0][0], 1.3345321e+03, delta=tolerance)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['pressure']['data'][2][0][0], 1.01328064e+03, delta=6.0E-05)
        self.assertAlmostEqual( result[5]['pressure']['data'][2][0][0], 1.97043777e+02, delta=2.0E-05)

        # Test derived equation for a single integration point
        result = self.mili.query("pressure", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['pressure']['data'][0][0][0], 1.23563398e+04, delta=5.0E-5)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("pressure", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        np.testing.assert_allclose( result[6]['pressure']['data'][0,:,:], PRESSURE )

    def test_disp_x(self):
        """X Displacement"""
        result = self.mili.query("disp_x", "node", states=[1,50,101], labels=[24,52])
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_x']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result[3]['disp_x']['data'][0][0][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_x']['data'][1][0][0], 1.00731432e-01)
        self.assertAlmostEqual( result[3]['disp_x']['data'][1][0][0], 0.000000e+00)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_x']['data'][2][0][0], 1.05899870e-01)
        self.assertAlmostEqual( result[3]['disp_x']['data'][2][0][0], 0.000000e+00)

    def test_disp_y(self):
        """Y Displacement"""
        result = self.mili.query("disp_y", "node", states=[1,50,101], labels=[24,52])
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_y']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result[3]['disp_y']['data'][0][0][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_y']['data'][1][0][0], 5.82261086e-02)
        self.assertAlmostEqual( result[3]['disp_y']['data'][1][0][0], 3.63588333e-06, delta=1e-14)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_y']['data'][2][0][0], 6.11518621e-02)
        self.assertAlmostEqual( result[3]['disp_y']['data'][2][0][0], 2.37822533e-05, delta=1e-13)

    def test_disp_y_reference_state(self):
        """Displacement with specified reference state."""
        result = self.mili.query("disp_y", "node", states=[50], labels=[24,52], reference_state=40)
        self.assertAlmostEqual( result[7]['disp_y']['data'][0][0][0], 1.73478723e-02)
        self.assertAlmostEqual( result[3]['disp_y']['data'][0][0][0], -3.99351120e-05, delta=1e-15)

    def test_disp_z(self):
        """Z Displacement"""
        result = self.mili.query("disp_z", "node", states=[1,50,101], labels=[24,52])
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_z']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result[3]['disp_z']['data'][0][0][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_z']['data'][1][0][0], -6.73872232e-02)
        self.assertAlmostEqual( result[3]['disp_z']['data'][1][0][0], -1.50992393e-01)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_z']['data'][2][0][0], -6.87063932e-02)
        self.assertAlmostEqual( result[3]['disp_z']['data'][2][0][0], -1.55236006e-01)

    def test_disp_mag(self):
        """Displacement Magnitude"""
        result = self.mili.query("disp_mag", "node", states=[1,50,101], labels=[24,52])
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][0][0][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][1][0][0], 1.34454966e-01)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][1][0][0], 1.50992393e-01)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][2][0][0], 1.40267283e-01)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][2][0][0], 1.55236006e-01)

    def test_disp_mag_ref(self):
        """Displacement Magnitude with reference state"""
        result = self.mili.query("disp_mag", "node", states=[1,50,101], labels=[24,52], reference_state=35)
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][0][0][0], 7.08499402e-02)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][0][0][0], 1.03274584e-01)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][1][0][0], 6.91310763e-02)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][1][0][0], 4.77178134e-02)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_mag']['data'][2][0][0], 7.52134994e-02)
        self.assertAlmostEqual( result[3]['disp_mag']['data'][2][0][0], 5.19614369e-02)

    def test_disp_rad_mag_xy(self):
        """Radial (XY) Displacement Magnitude"""
        result = self.mili.query("disp_rad_mag_xy", "node", states=[1,50,101], labels=[24,52])
        # State 1, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_rad_mag_xy']['data'][0][0][0], 0.000000e+00)
        self.assertAlmostEqual( result[3]['disp_rad_mag_xy']['data'][0][0][0], 0.000000e+00)
        # State 50, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_rad_mag_xy']['data'][1][0][0], 1.16349049e-01, delta=1e-08)
        self.assertAlmostEqual( result[3]['disp_rad_mag_xy']['data'][1][0][0], 3.63588333e-06, delta=1e-13)
        # State 101, nodes 24, 52
        self.assertAlmostEqual( result[7]['disp_rad_mag_xy']['data'][2][0][0], 1.22287907e-01, delta=1e-08)
        self.assertAlmostEqual( result[3]['disp_rad_mag_xy']['data'][2][0][0], 2.37822533e-05, delta=1e-12)

    def test_vol_strain(self):
        """Volumetric Strain"""
        result = self.mili.query("vol_strain", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['vol_strain']['data'][0][0][0], -1.53583100e-18, delta=5.0E-28)
        self.assertAlmostEqual( result[5]['vol_strain']['data'][0][0][0], 3.03314620e-18, delta=5.0E-27)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['vol_strain']['data'][1][0][0], -3.65868345e-06, delta=4.0E-15)
        self.assertAlmostEqual( result[5]['vol_strain']['data'][1][0][0], -5.33812863e-06, delta=7.0E-16)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['vol_strain']['data'][2][0][0], -4.05312221e-06, delta=2.0E-15)
        self.assertAlmostEqual( result[5]['vol_strain']['data'][2][0][0], -7.88175157e-07, delta=2.0E-16)

        # Test derived equation for a single integration point
        result = self.mili.query("vol_strain", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['vol_strain']['data'][0][0][0], -4.16194525e-05, delta=1e-13)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("vol_strain", "shell", labels=[4], states=[44], ips=[1,2])
        VOL_STRAIN = np.array([ [-3.15418220e-05, -4.16194525e-05] ])
        np.testing.assert_allclose( result[3]['vol_strain']['data'][0,:,:], VOL_STRAIN)

    def test_prin_strain1(self):
        """1st Principal Strain - This test fails. """
        result = self.mili.query("prin_strain1", "brick", states=[2,44,86], labels=[10, 20])
        # NOTE: For state 2 the answers look different when visually compared because the numbers are very small
        #       which causes the differences in floating point math between Griz (C) and python to be more pronounced.
        self.assertAlmostEqual( result[0]['prin_strain1']['data'][0][0][0], -5.11943684e-19, delta=7.0E-19)
        self.assertAlmostEqual( result[5]['prin_strain1']['data'][0][0][0], 1.01104870e-18, delta=3.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain1']['data'][1][0][0], 2.77111781e-06, delta=3.0E-13)
        self.assertAlmostEqual( result[5]['prin_strain1']['data'][1][0][0], 2.17467095e-05, delta=4.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain1']['data'][2][0][0], 4.38295842e-07, delta=6.0E-14)
        self.assertAlmostEqual( result[5]['prin_strain1']['data'][2][0][0], 2.66599636e-06, delta=3.0E-13)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain1", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_strain1']['data'][0][0][0], 1.52852561e-04, delta=2E-11)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain1", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_STRAIN1 = np.array([ [9.27323708e-05, 1.52852561e-04] ])
        np.testing.assert_allclose( result[3]['prin_strain1']['data'][0,:,:], PRIN_STRAIN1, atol=1E-20)

    def test_prin_strain2(self):
        """2nd Principal Strain """
        # Brick tests not yet completed (problems with brick strain results)
        result = self.mili.query("prin_strain2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain2']['data'][0][0][0], -5.11943684e-19, delta=6.0E-19)
        self.assertAlmostEqual( result[5]['prin_strain2']['data'][0][0][0], 1.01104870e-18, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain2']['data'][1][0][0], -2.66453526e-13, delta=4.0E-13)
        self.assertAlmostEqual( result[5]['prin_strain2']['data'][1][0][0], 2.81440538e-09, delta=4.0E-13)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain2']['data'][2][0][0], 1.15107923e-12, delta=2.0E-13)
        self.assertAlmostEqual( result[5]['prin_strain2']['data'][2][0][0], 8.12790990e-09, delta=2.0E-13)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain2", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_strain2']['data'][0][0][0], -5.42686757e-06, delta=7E-12)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain2", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_STRAIN2 = np.array([ [4.65712728e-05, -5.42686757e-06] ])
        np.testing.assert_allclose( result[3]['prin_strain2']['data'][0,:,:], PRIN_STRAIN2, atol=7E-12)

    def test_prin_strain3(self):
        """3rd Principal Strain """
        # Brick test values not yet set (problems with brick strain results)
        result = self.mili.query("prin_strain3", "brick", states=[2,44,86], labels=[10, 20])
        # NOTE: For state 2 the answers look different when visually compared because the numbers are very small
        #       which causes the differences in floating point math between Griz (C) and python to be more pronounced.
        self.assertAlmostEqual( result[0]['prin_strain3']['data'][0][0][0], -5.11943684e-19, delta=2.0E-18)
        self.assertAlmostEqual( result[5]['prin_strain3']['data'][0][0][0], 1.01104870e-18, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain3']['data'][1][0][0], -6.42980149e-06, delta=3.0E-15)
        self.assertAlmostEqual( result[5]['prin_strain3']['data'][1][0][0], -2.70876499e-05, delta=2.0E-12)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_strain3']['data'][2][0][0], -4.49141908e-06, delta=5.0E-13)
        self.assertAlmostEqual( result[5]['prin_strain3']['data'][2][0][0], -3.46229967e-06, delta=3.0E-13)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_strain3", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_strain3']['data'][0][0][0], -1.89045168e-04, delta=4E-13)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_strain3", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_STRAIN3 = np.array([ [-1.70845480e-04, -1.89045168e-04] ])
        np.testing.assert_allclose( result[3]['prin_strain3']['data'][0,:,:], PRIN_STRAIN3, atol=4E-13)

    def test_prin_dev_strain1(self):
        """1st Principal Deviatoric Strain """
        result = self.mili.query("prin_dev_strain1", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain1']['data'][0][0][0], 0.000000000e+00, delta=7.0E-19)
        self.assertAlmostEqual( result[5]['prin_dev_strain1']['data'][0][0][0], 0.000000000e+00, delta=3.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain1']['data'][1][0][0], 3.99067903e-06, delta=4.0E-15)
        self.assertAlmostEqual( result[5]['prin_dev_strain1']['data'][1][0][0], 2.35260850e-05, delta=2.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain1']['data'][2][0][0], 1.78933658e-06, delta=2.0E-13)
        self.assertAlmostEqual( result[5]['prin_dev_strain1']['data'][2][0][0], 2.92872141e-06, delta=3.0E-13)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_strain1", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_dev_strain1']['data'][0][0][0], 1.66725717e-04, delta=2E-11)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_strain1", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_DEV_STRAIN1 = np.array([ [1.03246311e-04, 1.66725717e-04] ])
        np.testing.assert_allclose( result[3]['prin_dev_strain1']['data'][0,:,:], PRIN_DEV_STRAIN1, atol=2E-11)

    def test_prin_dev_strain2(self):
        """2nd Principal Deviatoric Strain """
        result = self.mili.query("prin_dev_strain2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain2']['data'][0][0][0], 0.000000000e+00, delta=6.0E-19)
        self.assertAlmostEqual( result[5]['prin_dev_strain2']['data'][0][0][0], 0.000000000e+00, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain2']['data'][1][0][0], 1.21956089e-06, delta=4.0E-13)
        self.assertAlmostEqual( result[5]['prin_dev_strain2']['data'][1][0][0], 1.78219057e-06, delta=4.0E-13)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain2']['data'][2][0][0], 1.35104187e-06, delta=3.0E-13)
        self.assertAlmostEqual( result[5]['prin_dev_strain2']['data'][2][0][0], 2.70852951e-07, delta=2.0E-13)

        # Test derived equation for a single integration point - FAILS
        result = self.mili.query("prin_dev_strain2", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_dev_strain2']['data'][0][0][0], 8.44628357e-06, delta=8.0E-12)

        # Test derived equation when there are multiple integration points - FAILS
        result = self.mili.query("prin_dev_strain2", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_DEV_STRAIN2 = np.array([ [5.70852135e-05, 8.44628357e-06] ])
        np.testing.assert_allclose( result[3]['prin_dev_strain2']['data'][0,:,:], PRIN_DEV_STRAIN2, atol=8.0E-12)

    def test_prin_dev_strain3(self):
        """3rd Principal Deviatoric Strain """
        # Brick test values not yet set (problems with brick strain results)
        result = self.mili.query("prin_dev_strain3", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain3']['data'][0][0][0], 0.000000000e+00, delta=2.0E-18)
        self.assertAlmostEqual( result[5]['prin_dev_strain3']['data'][0][0][0], 0.000000000e+00, delta=2.0E-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain3']['data'][1][0][0], -5.21024049e-06, delta=5.0E-13)
        self.assertAlmostEqual( result[5]['prin_dev_strain3']['data'][1][0][0], -2.53082744e-05, delta=2.0E-14)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_strain3']['data'][2][0][0], -3.14037857e-06, delta=4.0E-15)
        self.assertAlmostEqual( result[5]['prin_dev_strain3']['data'][2][0][0], -3.19957462e-06, delta=5.0E-15)

        # Test derived equation for a single integration point - FAILS
        result = self.mili.query("prin_dev_strain3", "shell", labels=[4], states=[44], ips=[2])
        self.assertAlmostEqual( result[3]['prin_dev_strain3']['data'][0][0][0], -1.75172012e-04, delta=5E-14)

        # Test derived equation when there are multiple integration points - FAILS
        result = self.mili.query("prin_dev_strain3", "shell", labels=[4], states=[44], ips=[1,2])
        PRIN_DEV_STRAIN3 = np.array([ [-1.60331532e-4, -1.75172012e-04] ])
        np.testing.assert_allclose( result[3]['prin_dev_strain3']['data'][0,:,:], PRIN_DEV_STRAIN3, atol=5E-14)

    def test_prin_stress1(self):
        """1st Principal Stress"""
        result = self.mili.query("prin_stress1", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress1']['data'][0][0][0], -2.43397912e-10, delta=1e-16)
        self.assertAlmostEqual( result[5]['prin_stress1']['data'][0][0][0], 1.26490828e-09, delta=1e-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress1']['data'][1][0][0], 6.25506591e+00, delta=2.0E-5)
        self.assertAlmostEqual( result[5]['prin_stress1']['data'][1][0][0], 4.09456445e+03, delta=4.0E-6)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress1']['data'][2][0][0], -6.00356689e+02, delta=5.0E-7)
        self.assertAlmostEqual( result[5]['prin_stress1']['data'][2][0][0], 4.78815033e+02, delta=3.5E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress1", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['prin_stress1']['data'][0][0][0], 1.65892188e+03, delta=5.0E-3)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress1", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRIN_STRESS1 = np.array([ [1.23608057e+03, 1.65892188e+03, 1.73505859e+03, 1.32386670e+03] ])
        np.testing.assert_allclose( result[6]['prin_stress1']['data'][0,:,:], PRIN_STRESS1, rtol=4.0E-06)

    def test_prin_stress2(self):
        """2nd Principal Stress"""
        result = self.mili.query("prin_stress2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress2']['data'][0][0][0], -2.65816896e-10)
        self.assertAlmostEqual( result[5]['prin_stress2']['data'][0][0][0], 5.24967625e-10)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress2']['data'][1][0][0], -6.33233765e+02, delta=2.0e-4)
        self.assertAlmostEqual( result[5]['prin_stress2']['data'][1][0][0], -9.23257324e+02, delta=1.5E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress2']['data'][2][0][0], -7.01501648e+02, delta=1.0E-4)
        self.assertAlmostEqual( result[5]['prin_stress2']['data'][2][0][0], -1.34539261e+02, delta=3.5E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress2", "shell", labels=[2,11], states=[44], ips=[1])
        self.assertEqual(result[3]['prin_stress2']['source'], 'derived')
        self.assertAlmostEqual( result[3]['prin_stress2']['data'][0][0][0], 1.24129639e+03, delta=4.0E-06)
        self.assertAlmostEqual( result[2]['prin_stress2']['data'][0][0][0], -8.73693164e+03, delta=1.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress2", "shell", labels=[2], states=[44], ips=[1,2])
        self.assertEqual(result[3]['prin_stress2']['source'], 'derived')
        PRIN_STRESS2 = np.array([ [1.24129639e+03, -2.53508789e+03] ])
        np.testing.assert_allclose( result[3]['prin_stress2']['data'][0,:,:], PRIN_STRESS2)

    def test_prin_stress3(self):
        """3rd Principal Stress"""
        result = self.mili.query("prin_stress3", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress3']['data'][0][0][0], -6.42658482e-10, delta=1e-19)
        self.assertAlmostEqual( result[5]['prin_stress3']['data'][0][0][0], 4.84983775e-10, delta=1e-16)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress3']['data'][1][0][0], -2.11703394e+03, delta=5.0E-6)
        self.assertAlmostEqual( result[5]['prin_stress3']['data'][1][0][0], -7.17490381e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_stress3']['data'][2][0][0], -1.73798328e+03, delta=2.0E-4)
        self.assertAlmostEqual( result[5]['prin_stress3']['data'][2][0][0], -9.35407166e+02, delta=1.0E-4)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_stress3", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['prin_stress3']['data'][0][0][0], -3.87279453e+04, delta=2.0E-5)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_stress3", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRIN_STRESS3 = np.array([ [-2.40547637e+04, -3.87279453e+04, -3.81615352e+04, -2.34779512e+04] ])
        np.testing.assert_allclose( result[6]['prin_stress3']['data'][0,:,:], PRIN_STRESS3, rtol=2.0E-07)

    def test_eff_stress(self):
        """Effective (von Mises) Stress"""
        result = self.mili.query("eff_stress", "brick", states=[2,44,86], labels=[10, 20])

        self.assertEqual(result[0]['eff_stress']['source'], 'derived')
        self.assertEqual(result[0]['eff_stress']['class_name'], 'brick')
        self.assertEqual(result[0]['eff_stress']['title'], 'Effective Stress')
        self.assertEqual(result[0]['eff_stress']['layout']['components'], ['eff_stress'])
        np.testing.assert_equal(result[0]['eff_stress']['layout']['labels'], [10])
        np.testing.assert_allclose(result[0]['eff_stress']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result[0]['eff_stress']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )
        self.assertEqual(result[5]['eff_stress']['source'], 'derived')
        self.assertEqual(result[5]['eff_stress']['class_name'], 'brick')
        self.assertEqual(result[5]['eff_stress']['title'], 'Effective Stress')
        self.assertEqual(result[5]['eff_stress']['layout']['components'], ['eff_stress'])
        np.testing.assert_equal(result[5]['eff_stress']['layout']['labels'], [20])
        np.testing.assert_allclose(result[5]['eff_stress']['layout']['states'], [2,44,86] )
        np.testing.assert_allclose(result[5]['eff_stress']['layout']['times'], [1.0e-05, 4.3e-04, 8.5e-04] )

        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['eff_stress']['data'][0][0][0], 3.88536481e-10, delta=1e-18)
        self.assertAlmostEqual( result[5]['eff_stress']['data'][0][0][0], 7.60721042e-10, delta=1e-18)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['eff_stress']['data'][1][0][0], 1.88665918e+03, delta=3.5E-7)
        self.assertAlmostEqual( result[5]['eff_stress']['data'][1][0][0], 9.77912305e+03, delta=3.5E-6)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['eff_stress']['data'][2][0][0], 1.09057764e+03, delta=3.5E-6)
        self.assertAlmostEqual( result[5]['eff_stress']['data'][2][0][0], 1.22833582e+03, delta=5.0E-6)

        # Test derived equation for a single integration point
        result = self.mili.query("eff_stress", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['eff_stress']['data'][0][0][0], 3.95834883e+04, delta=2.0E-05)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("eff_stress", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        EFF_STRESS = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        np.testing.assert_allclose( result[6]['eff_stress']['data'][0,:,:], EFF_STRESS)

    def test_prin_dev_stress1(self):
        """1st Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress1", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress1']['data'][0][0][0], 1.40559855e-10, delta=2.0E-17)
        self.assertAlmostEqual( result[5]['prin_dev_stress1']['data'][0][0][0], 5.06621689e-10, delta=6.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress1']['data'][1][0][0], 9.20925903e+02, delta=3.5E-7)
        self.assertAlmostEqual( result[5]['prin_dev_stress1']['data'][1][0][0], 5.42909668e+03, delta=3.5E-7)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress1']['data'][2][0][0], 4.12923828e+02, delta=1.5E-4)
        self.assertAlmostEqual( result[5]['prin_dev_stress1']['data'][2][0][0], 6.75858826e+02, delta=3.5E-7)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress1", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['prin_dev_stress1']['data'][0][0][0], 1.40152617e+04, delta=5.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress1", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRIN_DEV_STRESS1 = np.array([ [8.84230957e+03, 1.40152617e+04, 1.38772188e+04, 8.70856152e+03] ])
        np.testing.assert_allclose( result[6]['prin_dev_stress1']['data'][0,:,:], PRIN_DEV_STRESS1, rtol=5.0E-07)

    def test_prin_dev_stress2(self):
        """2nd Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress2", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress2']['data'][0][0][0], 1.18140858e-10, delta=2.5E-19)
        self.assertAlmostEqual( result[5]['prin_dev_stress2']['data'][0][0][0], -2.33318975e-10, delta=6.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress2']['data'][1][0][0], 2.81437103e+02, delta=6.5E-5)
        self.assertAlmostEqual( result[5]['prin_dev_stress2']['data'][1][0][0], 4.11274780e+02, delta=1.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress2']['data'][2][0][0], 3.11778900e+02, delta=1.5E-4)
        self.assertAlmostEqual( result[5]['prin_dev_stress2']['data'][2][0][0], 6.25045280e+01, delta=4.0E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress2", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['prin_dev_stress2']['data'][0][0][0], 1.23563418e+04, delta=2.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress2", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRIN_DEV_STRESS2 = np.array([ [7.60622510e+03, 1.23563418e+04, 1.21421543e+04, 7.38469531e+03] ])
        np.testing.assert_allclose( result[6]['prin_dev_stress2']['data'][0,:,:], PRIN_DEV_STRESS2, rtol=6.0E-07)

    def test_prin_dev_stress3(self):
        """3rd Principal Deviatoric Stress"""
        result = self.mili.query("prin_dev_stress3", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress3']['data'][0][0][0], -2.58700728e-10, delta=5.0E-19)
        self.assertAlmostEqual( result[5]['prin_dev_stress3']['data'][0][0][0], -2.73302797e-10, delta=3.0E-17)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress3']['data'][1][0][0], -1.20236304e+03, delta=1.5E-4)
        self.assertAlmostEqual( result[5]['prin_dev_stress3']['data'][1][0][0], -5.84037158e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['prin_dev_stress3']['data'][2][0][0], -7.24702759e+02, delta=1.5E-4)
        self.assertAlmostEqual( result[5]['prin_dev_stress3']['data'][2][0][0], -7.38363403e+02, delta=6.5E-5)

        # Test derived equation for a single integration point
        result = self.mili.query("prin_dev_stress3", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['prin_dev_stress3']['data'][0][0][0], -2.63716055e+04, delta=4.0E-05)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("prin_dev_stress3", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRIN_DEV_STRESS3 = np.array([ [-1.64485352e+04, -2.63716055e+04, -2.60193750e+04, -1.60932568e+04] ])
        np.testing.assert_allclose( result[6]['prin_dev_stress3']['data'][0,:,:], PRIN_DEV_STRESS3, rtol=2.0E-07)

    def test_max_shear_stress(self):
        """Max. Shear Stress"""
        result = self.mili.query("max_shear_stress", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['max_shear_stress']['data'][0][0][0], 1.99630285e-10, delta=1.5E-17)
        self.assertAlmostEqual( result[5]['max_shear_stress']['data'][0][0][0], 3.89962229e-10, delta=2.0E-19)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['max_shear_stress']['data'][1][0][0], 1.06164453e+03, delta=1.5E-6)
        self.assertAlmostEqual( result[5]['max_shear_stress']['data'][1][0][0], 5.63473438e+03, delta=5.0E-4)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['max_shear_stress']['data'][2][0][0], 5.68813293e+02, delta=6.5E-5)
        self.assertAlmostEqual( result[5]['max_shear_stress']['data'][2][0][0], 7.07111084e+02, delta=5.5E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("max_shear_stress", "beam", labels=[20], states=[44], ips=[2])
        self.assertAlmostEqual( result[6]['max_shear_stress']['data'][0][0][0], 2.01934336e+04, delta=2.0E-03)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("max_shear_stress", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        MAX_SHEAR_STRESS = np.array([ [1.26454219e+04, 2.01934336e+04, 1.99482969e+04, 1.24009092e+04] ])
        np.testing.assert_allclose( result[6]['max_shear_stress']['data'][0,:,:], MAX_SHEAR_STRESS, rtol=2.0E-07)

    def test_triaxiality(self):
        """Triaxiality"""
        # Answers are based on hand calculations using griz results for pressure and seff.
        result = self.mili.query("triaxiality", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['triaxiality']['data'][0][0][0], -3.83957754/3.88536481, delta=1.0E-08)
        self.assertAlmostEqual( result[5]['triaxiality']['data'][0][0][0], +7.58286600/7.60721042, delta=1.0E-06)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['triaxiality']['data'][1][0][0], -9.14670837e+02/1.88665918e+03, delta=1.0E-8)
        self.assertAlmostEqual( result[5]['triaxiality']['data'][1][0][0], -1.33453210e+03/9.77912305e+03, delta=1.0E-8)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['triaxiality']['data'][2][0][0], -1.01328064e+03/1.0905776e+03, delta=6.0E-8)
        self.assertAlmostEqual( result[5]['triaxiality']['data'][2][0][0], -1.970438e+02/1.2283358e+03, delta=1.0E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("triaxiality", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result[6]['triaxiality']['source'], 'derived')
        self.assertAlmostEqual( result[6]['triaxiality']['data'][0][0][0], -1.23563398e+04/3.95834883e+04, delta=1.0E-08)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("triaxiality", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        SEFF = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        TRIAXIALITY = -PRESSURE/SEFF
        np.testing.assert_allclose( result[6]['triaxiality']['data'][0,:,:], TRIAXIALITY, rtol=1.0E-07)

    def test_normalized_pressure(self):
        """Normalized Pressure"""
        # Answers are based on hand calculations using griz results for pressure and seff.
        result = self.mili.query("norm_press", "brick", states=[2,44,86], labels=[10, 20])
        # State 2, labels 10, 20
        self.assertAlmostEqual( result[0]['norm_press']['data'][0][0][0], +3.83957754/3.88536481, delta=1.0E-08)
        self.assertAlmostEqual( result[5]['norm_press']['data'][0][0][0], -7.58286600/7.60721042, delta=1.0E-06)
        # State 44, labels 10, 20
        self.assertAlmostEqual( result[0]['norm_press']['data'][1][0][0], +9.14670837e+02/1.88665918e+03, delta=1.0E-8)
        self.assertAlmostEqual( result[5]['norm_press']['data'][1][0][0], +1.33453210e+03/9.77912305e+03, delta=1.0E-8)
        # State 86, labels 10, 20
        self.assertAlmostEqual( result[0]['norm_press']['data'][2][0][0], +1.01328064e+03/1.0905776e+03, delta=6.0E-8)
        self.assertAlmostEqual( result[5]['norm_press']['data'][2][0][0], +1.970438e+02/1.2283358e+03, delta=1.0E-8)

        # Test derived equation for a single integration point
        result = self.mili.query("norm_press", "beam", labels=[20], states=[44], ips=[2])
        self.assertEqual(result[6]['norm_press']['source'], 'derived')
        self.assertAlmostEqual( result[6]['norm_press']['data'][0][0][0], +1.23563398e+04/3.95834883e+04, delta=1.0E-08)

        # Test derived equation when there are multiple integration points
        result = self.mili.query("norm_press", "beam", labels=[20], states=[44], ips=[1,2,3,4])
        PRESSURE = np.array([ [7.60622900e+03, 1.23563398e+04, 1.21421602e+04, 7.38469482e+03] ])
        SEFF = np.array([ [2.46960137e+04, 3.95834883e+04, 3.90579766e+04, 2.41670957e+04] ])
        NORMALIZED_PRESSURE = PRESSURE/SEFF
        np.testing.assert_allclose( result[6]['norm_press']['data'][0,:,:], NORMALIZED_PRESSURE, rtol=1.0E-07)

    def test_hex_element_volume(self):
        """Test Element Volume calculation for Hexes."""
        result = self.mili.query("element_volume", "brick", labels=[1,4], states=[1,3,4])

        np.testing.assert_equal( result[3]["element_volume"]["layout"]["labels"], [1,4])
        np.testing.assert_equal( result[3]["element_volume"]["layout"]["states"], [1,3,4])
        np.testing.assert_allclose( result[3]["element_volume"]["layout"]["times"], [0.0e00, 2.0e-05, 3.0e-05])
        self.assertEqual( result[3]["element_volume"]["source"], "derived")
        self.assertEqual( result[3]["element_volume"]["class_name"], "brick")
        self.assertEqual( result[3]["element_volume"]["title"], "Element Volume")

        self.assertAlmostEqual(result[3]["element_volume"]["data"][0,0,0], 0.02083334)
        self.assertAlmostEqual(result[3]["element_volume"]["data"][0,1,0], 0.0291667)
        self.assertAlmostEqual(result[3]["element_volume"]["data"][1,0,0], 0.02083334)
        self.assertAlmostEqual(result[3]["element_volume"]["data"][1,1,0], 0.0291667)
        self.assertAlmostEqual(result[3]["element_volume"]["data"][2,0,0], 0.02083334)
        self.assertAlmostEqual(result[3]["element_volume"]["data"][2,1,0], 0.0291667)

        result = combine( self.mili.query("element_volume", "brick", labels=[1,4], states=[1,3,4]) )

        np.testing.assert_equal( result["element_volume"]["layout"]["labels"], [1,4])
        np.testing.assert_equal( result["element_volume"]["layout"]["states"], [1,3,4])
        np.testing.assert_allclose( result["element_volume"]["layout"]["times"], [0.0e00, 2.0e-05, 3.0e-05])
        self.assertEqual( result["element_volume"]["source"], "derived")
        self.assertEqual( result["element_volume"]["class_name"], "brick")
        self.assertEqual( result["element_volume"]["title"], "Element Volume")

        self.assertAlmostEqual(result["element_volume"]["data"][0,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][0,1,0], 0.0291667)
        self.assertAlmostEqual(result["element_volume"]["data"][1,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][1,1,0], 0.0291667)
        self.assertAlmostEqual(result["element_volume"]["data"][2,0,0], 0.02083334)
        self.assertAlmostEqual(result["element_volume"]["data"][2,1,0], 0.0291667)

    def test_quad_area(self):
        """Test area calculation for quad elements."""
        result = combine( self.mili.query("area", "cseg", labels=[1,12,24], states=[1,40,80]) )

        np.testing.assert_equal( result["area"]["layout"]["labels"], [12,24,1])
        np.testing.assert_equal( result["area"]["layout"]["states"], [1,40,80])
        np.testing.assert_allclose( result["area"]["layout"]["times"], [0.0,  0.00039, 0.00079])
        self.assertEqual( result["area"]["source"], "derived")
        self.assertEqual( result["area"]["class_name"], "cseg")
        self.assertEqual( result["area"]["title"], "Quad Area")

        # State 1
        self.assertAlmostEqual(result["area"]["data"][0,0,0], 0.171875)
        self.assertAlmostEqual(result["area"]["data"][0,1,0], 0.17187496)
        self.assertAlmostEqual(result["area"]["data"][0,2,0], 0.078125)
        # State 40
        self.assertAlmostEqual(result["area"]["data"][1,0,0], 0.17187835)
        self.assertAlmostEqual(result["area"]["data"][1,1,0], 0.17187496)
        self.assertAlmostEqual(result["area"]["data"][1,2,0], 0.07812971)
        # State 80
        self.assertAlmostEqual(result["area"]["data"][2,0,0], 0.17187445)
        self.assertAlmostEqual(result["area"]["data"][2,1,0], 0.17187496)
        self.assertAlmostEqual(result["area"]["data"][2,2,0], 0.07812706)

    def test_surfstrain(self):
        """Test derived result surfstrain."""
        file_name = os.path.join(dir_path,'data','parallel','basic1','basic1.plt')
        db = open_database( file_name, suppress_parallel=True )

        with self.assertRaises(MiliPythonError):
            # Missing face number
            result = db.query("surfstrainz", "brick", labels=[151,229], states=[2,20,40])
        with self.assertRaises(MiliPythonError):
            # Face number is invalid
            result = db.query("surfstrainz", "brick", labels=[151,229], states=[2,20,40], face=7)

        result = db.query("surfstrainz", "brick", labels=[151,229], states=[2,20,40], face=2)
        # State 21
        self.assertAlmostEqual(result["surfstrainz"]["data"][0,0,0], -1.1443692e-05)
        self.assertAlmostEqual(result["surfstrainz"]["data"][0,1,0], 1.4395356e-06)
        # State 22
        self.assertAlmostEqual(result["surfstrainz"]["data"][1,0,0], -3.4624312e-04)
        self.assertAlmostEqual(result["surfstrainz"]["data"][1,1,0], 1.6831054e-05)
        # State 23
        self.assertAlmostEqual(result["surfstrainz"]["data"][2,0,0], -4.2545883e-04)
        self.assertAlmostEqual(result["surfstrainz"]["data"][2,1,0], -4.0996027e-05)

        with self.assertRaises(MiliPythonError):
            # Missing face number
            result = db.query("surfstrainxy", "brick", labels=[151,229], states=[2,20,40])
        with self.assertRaises(MiliPythonError):
            # Face number is invalid
            result = db.query("surfstrainxy", "brick", labels=[151,229], states=[2,20,40], face=7)

        result = db.query("surfstrainxy", "brick", labels=[151,229], states=[2,20,40], face=5)
        # State 21
        self.assertAlmostEqual(result["surfstrainxy"]["data"][0,0,0], -1.1861324e-05)
        self.assertAlmostEqual(result["surfstrainxy"]["data"][0,1,0], -1.0636501e-06)
        # State 22
        self.assertAlmostEqual(result["surfstrainxy"]["data"][1,0,0], -2.0110358e-04)
        self.assertAlmostEqual(result["surfstrainxy"]["data"][1,1,0], -4.2742749e-06)
        # State 23
        self.assertAlmostEqual(result["surfstrainxy"]["data"][2,0,0], -2.1373070e-04)
        self.assertAlmostEqual(result["surfstrainxy"]["data"][2,1,0], -4.2441879e-06)

        db.close()