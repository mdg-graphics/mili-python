#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import os
import unittest
from mili import miliinternal
from mili.miliinternal import *
from mili.adjacency import GeometricMeshInfo
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestMiliInternal(unittest.TestCase):
    """Tests the _MiliInternal class using the d3samp6 database.

    A subset of functions are tested elsewhere:
    - query
    - append_state -> test_append_states.py
    - copy_non_state_data -> test_append_states.py
    """
    dir_name = os.path.join(dir_path,'data','serial','sstate')
    basename = 'd3samp6.plt'

    def setUp(self):
        self.mili = miliinternal._MiliInternal(TestMiliInternal.dir_name, TestMiliInternal.basename)

    #==============================================================================
    def test_geometry_property(self):
        geom = self.mili.geometry
        self.assertTrue(isinstance(geom, GeometricMeshInfo))

    #==============================================================================
    def test_clean_return_code(self):
        # Just check that it runs without errors
        self.mili.clear_return_code()

    #==============================================================================
    def test_returncode(self):
        ret_code = self.mili.returncode()
        np.testing.assert_equal(ret_code, (ReturnCode.OK, ""))

    #==============================================================================
    def test_reload_state_maps(self):
        # Just check that it runs without errors
        self.mili.reload_state_maps()

    #==============================================================================
    def test_nodes(self):
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

    #==============================================================================
    def test_state_maps(self):
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        state_maps = self.mili.state_maps()
        self.assertEqual(STATE_COUNT, len(state_maps))
        self.assertEqual(FIRST_STATE, state_maps[0].time)
        self.assertEqual(LAST_STATE, state_maps[-1].time)

    #==============================================================================
    def test_subrecords(self):
        srecs = self.mili.subrecords()
        self.assertEqual(len(srecs), 23)
        for srec in srecs:
            self.assertTrue(isinstance(srec, Subrecord))

    #==============================================================================
    def test_parameters(self):
        params = self.mili.parameters()
        self.assertTrue(isinstance(params, dict))

    #==============================================================================
    def test_parameter(self):
        title = self.mili.parameter("title")
        self.assertEqual(title, "Beam element example (in,sec,lb-s^2/in)Quarter model INGDY6.dat")

        not_found = self.mili.parameter("does-not-exists")
        self.assertEqual(not_found, None)

        not_found = self.mili.parameter("does-not-exists", "not_found")
        self.assertEqual(not_found, "not_found")

    #==============================================================================
    def test_srec_fmt_qty(self):
        srec_fmt_qty = self.mili.srec_fmt_qty()
        self.assertEqual(srec_fmt_qty, 1)

    #==============================================================================
    def test_mesh_dimensions(self):
        dims = self.mili.mesh_dimensions()
        self.assertEqual(dims, 3)

    #==============================================================================
    def test_class_names(self):
        class_names = self.mili.class_names()
        self.assertEqual(class_names, ["glob", "mat", "node", "beam", "brick", "shell", "cseg"])

    #==============================================================================
    def test_mesh_object_classes(self):
        mo_classes = self.mili.mesh_object_classes()

        # Glob class
        glob_class = mo_classes["glob"]
        self.assertEqual(glob_class.mesh_id, 0)
        self.assertEqual(glob_class.short_name, "glob")
        self.assertEqual(glob_class.long_name, "Global")
        self.assertEqual(glob_class.sclass, Superclass.M_MESH)
        self.assertEqual(glob_class.elem_qty, 1)
        self.assertEqual(glob_class.idents_exist, True)

        # Mat Class
        mat_class = mo_classes["mat"]
        self.assertEqual(mat_class.mesh_id, 0)
        self.assertEqual(mat_class.short_name, "mat")
        self.assertEqual(mat_class.long_name, "Material")
        self.assertEqual(mat_class.sclass, Superclass.M_MAT)
        self.assertEqual(mat_class.elem_qty, 5)
        self.assertEqual(mat_class.idents_exist, True)

        # Node class
        node_class = mo_classes["node"]
        self.assertEqual(node_class.mesh_id, 0)
        self.assertEqual(node_class.short_name, "node")
        self.assertEqual(node_class.long_name, "Node")
        self.assertEqual(node_class.sclass, Superclass.M_NODE)
        self.assertEqual(node_class.elem_qty, 144)
        self.assertEqual(node_class.idents_exist, True)

        # beam class
        beam_class = mo_classes["beam"]
        self.assertEqual(beam_class.mesh_id, 0)
        self.assertEqual(beam_class.short_name, "beam")
        self.assertEqual(beam_class.long_name, "Beams")
        self.assertEqual(beam_class.sclass, Superclass.M_BEAM)
        self.assertEqual(beam_class.elem_qty, 46)
        self.assertEqual(beam_class.idents_exist, True)

        # brick class
        brick_class = mo_classes["brick"]
        self.assertEqual(brick_class.mesh_id, 0)
        self.assertEqual(brick_class.short_name, "brick")
        self.assertEqual(brick_class.long_name, "Bricks")
        self.assertEqual(brick_class.sclass, Superclass.M_HEX)
        self.assertEqual(brick_class.elem_qty, 36)
        self.assertEqual(brick_class.idents_exist, True)

        # shell class
        shell_class = mo_classes["shell"]
        self.assertEqual(shell_class.mesh_id, 0)
        self.assertEqual(shell_class.short_name, "shell")
        self.assertEqual(shell_class.long_name, "Shells")
        self.assertEqual(shell_class.sclass, Superclass.M_QUAD)
        self.assertEqual(shell_class.elem_qty, 12)
        self.assertEqual(shell_class.idents_exist, True)

        # cseg class
        cseg_class = mo_classes["cseg"]
        self.assertEqual(cseg_class.mesh_id, 0)
        self.assertEqual(cseg_class.short_name, "cseg")
        self.assertEqual(cseg_class.long_name, "Contact Segment")
        self.assertEqual(cseg_class.sclass, Superclass.M_QUAD)
        self.assertEqual(cseg_class.elem_qty, 24)

    #==============================================================================
    def test_int_points_of_state_variable(self):
        ipts = self.mili.int_points_of_state_variable("sx", "brick")
        np.testing.assert_equal(ipts, [])

        ipts = self.mili.int_points_of_state_variable("sx", "beam")
        np.testing.assert_equal(ipts, [1,2,3,4])

        ipts = self.mili.int_points_of_state_variable("sx", "shell")
        np.testing.assert_equal(ipts, [1,2])

    #==============================================================================
    def test_element_set(self):
        elem_sets = self.mili.element_sets()
        self.assertEqual(elem_sets, {"es_1": [1,2,3,4,4], "es_3": [1,2,2]})

    #==============================================================================
    def test_integration_points(self):
        ipts = self.mili.integration_points()
        self.assertEqual( ipts, { "1": [1,2,3,4], "3": [1,2]} )

    #==============================================================================
    def test_times(self):
        FIRST_STATE = 0.0
        LAST_STATE = 0.0010000000474974513
        STATE_COUNT = 101
        times = self.mili.times()
        self.assertEqual(STATE_COUNT, len(times))
        self.assertEqual(FIRST_STATE, times[0])
        self.assertEqual(LAST_STATE, times[-1])

        times = self.mili.times([0,100])
        self.assertEqual(FIRST_STATE, times[0])
        self.assertEqual(LAST_STATE, times[-1])

    #==============================================================================
    def test_state_variables(self):
        svars = self.mili.state_variables()
        self.assertEqual(len(svars), 156)
        for svar in svars.values():
            self.assertTrue(isinstance(svar, StateVariable))

    #==============================================================================
    def test_queriable_svars(self):
        svars = self.mili.queriable_svars()
        expected = ['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
                    'tcon_fric_eng', 'tcon_eng', 'ew', 'te', 'rbvx', 'rbvy', 'rbvz', 'rbax',
                    'rbay', 'rbaz', 'init', 'plot', 'hsp', 'other_i_o', 'brick', 'beam', 'shell',
                    'tshell', 'discrete', 'delam', 'cohesive', 'ml', 'ntet', 'sph', 'kin_contact',
                    'reglag_contact', 'lag_solver', 'coupling', 'solution', 'xfem', 'total', 'cpu_time',
                    'cpu_time[init]', 'cpu_time[plot]', 'cpu_time[hsp]', 'cpu_time[other_i_o]', 'cpu_time[brick]',
                    'cpu_time[beam]', 'cpu_time[shell]', 'cpu_time[tshell]', 'cpu_time[discrete]', 'cpu_time[delam]',
                    'cpu_time[cohesive]', 'cpu_time[ml]', 'cpu_time[ntet]', 'cpu_time[sph]', 'cpu_time[kin_contact]',
                    'cpu_time[reglag_contact]', 'cpu_time[lag_solver]', 'cpu_time[coupling]', 'cpu_time[solution]',
                    'cpu_time[xfem]', 'cpu_time[total]', 'matpe', 'matke', 'mathe', 'matbve', 'matdre', 'matstde',
                    'matflde', 'matte', 'matmass', 'matcgx', 'matcgy', 'matcgz', 'matxv', 'matyv', 'matzv', 'matxa',
                    'matya', 'matza', 'con_forx', 'con_fory', 'con_forz', 'con_momx', 'con_momy', 'con_momz',
                    'failure_bs', 'total_bs', 'cycles_bs', 'con_damp_eng', 'con_fric_eng', 'con_eng', 'sn', 'shmag',
                    'sr', 'ss', 's1', 's2', 's3', 'cseg_var', 'cseg_var[sn]', 'cseg_var[shmag]', 'cseg_var[sr]',
                    'cseg_var[ss]', 'cseg_var[s1]', 'cseg_var[s2]', 'cseg_var[s3]', 'ux', 'uy', 'uz', 'nodpos',
                    'nodpos[ux]', 'nodpos[uy]', 'nodpos[uz]', 'vx', 'vy', 'vz', 'nodvel', 'nodvel[vx]', 'nodvel[vy]',
                    'nodvel[vz]', 'ax', 'ay', 'az', 'nodacc', 'nodacc[ax]', 'nodacc[ay]', 'nodacc[az]', 'sx', 'sy',
                    'sz', 'sxy', 'syz', 'szx', 'stress', 'stress[sx]', 'stress[sy]', 'stress[sz]', 'stress[sxy]',
                    'stress[syz]', 'stress[szx]', 'eps', 'es_1a[sx]', 'es_1a[sy]', 'es_1a[sz]', 'es_1a[sxy]',
                    'es_1a[syz]', 'es_1a[szx]', 'es_1a[eps]', 'ex', 'ey', 'ez', 'exy', 'eyz', 'ezx', 'strain',
                    'strain[ex]', 'strain[ey]', 'strain[ez]', 'strain[exy]', 'strain[eyz]', 'strain[ezx]', 'edrate',
                    'es_3a[sx]', 'es_3a[sy]', 'es_3a[sz]', 'es_3a[sxy]', 'es_3a[syz]', 'es_3a[szx]', 'es_3c[ex]', 'es_3c[ey]',
                    'es_3c[ez]', 'es_3c[exy]', 'es_3c[eyz]', 'es_3c[ezx]', 'es_3c[edrate]', 'axf', 'sfs', 'sft', 'ms',
                    'mt', 'tor', 'max_eps', 'svec_x', 'svec_y', 'svec_z', 'svec', 'svec[svec_x]', 'svec[svec_y]', 'svec[svec_z]',
                    'efs1', 'efs2', 'eps1', 'eps2', 'stress_mid', 'stress_mid[sx]', 'stress_mid[sy]', 'stress_mid[sz]', 'stress_mid[sxy]',
                    'stress_mid[syz]', 'stress_mid[szx]', 'eeff_mid', 'stress_in', 'stress_in[sx]', 'stress_in[sy]', 'stress_in[sz]',
                    'stress_in[sxy]', 'stress_in[syz]', 'stress_in[szx]', 'eeff_in', 'stress_out', 'stress_out[sx]', 'stress_out[sy]',
                    'stress_out[sz]', 'stress_out[sxy]', 'stress_out[syz]', 'stress_out[szx]', 'eeff_out', 'mxx', 'myy', 'mxy', 'bend',
                    'bend[mxx]', 'bend[myy]', 'bend[mxy]', 'qxx', 'qyy', 'shear', 'shear[qxx]', 'shear[qyy]', 'nxx', 'nyy', 'nxy', 'normal',
                    'normal[nxx]', 'normal[nyy]', 'normal[nxy]', 'thick', 'edv1', 'edv2', 'inteng', 'mid', 'in', 'out', 'press_cut',
                    'press_cut[mid]', 'press_cut[in]', 'press_cut[out]', 'd_1', 'd_1[mid]', 'd_1[in]', 'd_1[out]', 'd_2', 'd_2[mid]',
                    'd_2[in]', 'd_2[out]', 'dam', 'dam[mid]', 'dam[in]', 'dam[out]', 'frac_strain', 'frac_strain[mid]', 'frac_strain[in]',
                    'frac_strain[out]', 'sand', 'cause']
        np.testing.assert_equal(svars, expected)

        svars = self.mili.queriable_svars(vector_only=True)
        expected = ['cpu_time', 'cpu_time[init]', 'cpu_time[plot]', 'cpu_time[hsp]', 'cpu_time[other_i_o]', 'cpu_time[brick]',
                    'cpu_time[beam]', 'cpu_time[shell]', 'cpu_time[tshell]', 'cpu_time[discrete]', 'cpu_time[delam]', 'cpu_time[cohesive]',
                    'cpu_time[ml]', 'cpu_time[ntet]', 'cpu_time[sph]', 'cpu_time[kin_contact]', 'cpu_time[reglag_contact]',
                    'cpu_time[lag_solver]', 'cpu_time[coupling]', 'cpu_time[solution]', 'cpu_time[xfem]', 'cpu_time[total]', 'cseg_var',
                    'cseg_var[sn]', 'cseg_var[shmag]', 'cseg_var[sr]', 'cseg_var[ss]', 'cseg_var[s1]', 'cseg_var[s2]', 'cseg_var[s3]',
                    'nodpos', 'nodpos[ux]', 'nodpos[uy]', 'nodpos[uz]', 'nodvel', 'nodvel[vx]', 'nodvel[vy]', 'nodvel[vz]', 'nodacc',
                    'nodacc[ax]', 'nodacc[ay]', 'nodacc[az]', 'stress', 'stress[sx]', 'stress[sy]', 'stress[sz]', 'stress[sxy]', 'stress[syz]',
                    'stress[szx]', 'es_1a[sx]', 'es_1a[sy]', 'es_1a[sz]', 'es_1a[sxy]', 'es_1a[syz]', 'es_1a[szx]', 'es_1a[eps]', 'strain',
                    'strain[ex]', 'strain[ey]', 'strain[ez]', 'strain[exy]', 'strain[eyz]', 'strain[ezx]', 'es_3a[sx]', 'es_3a[sy]', 'es_3a[sz]',
                    'es_3a[sxy]', 'es_3a[syz]', 'es_3a[szx]', 'es_3c[ex]', 'es_3c[ey]', 'es_3c[ez]', 'es_3c[exy]', 'es_3c[eyz]', 'es_3c[ezx]',
                    'es_3c[edrate]', 'svec', 'svec[svec_x]', 'svec[svec_y]', 'svec[svec_z]', 'stress_mid', 'stress_mid[sx]', 'stress_mid[sy]',
                    'stress_mid[sz]', 'stress_mid[sxy]', 'stress_mid[syz]', 'stress_mid[szx]', 'stress_in', 'stress_in[sx]', 'stress_in[sy]',
                    'stress_in[sz]', 'stress_in[sxy]', 'stress_in[syz]', 'stress_in[szx]', 'stress_out', 'stress_out[sx]', 'stress_out[sy]',
                    'stress_out[sz]', 'stress_out[sxy]', 'stress_out[syz]', 'stress_out[szx]', 'bend', 'bend[mxx]', 'bend[myy]', 'bend[mxy]',
                    'shear', 'shear[qxx]', 'shear[qyy]', 'normal', 'normal[nxx]', 'normal[nyy]', 'normal[nxy]', 'press_cut', 'press_cut[mid]',
                    'press_cut[in]', 'press_cut[out]', 'd_1', 'd_1[mid]', 'd_1[in]', 'd_1[out]', 'd_2', 'd_2[mid]', 'd_2[in]', 'd_2[out]', 'dam',
                    'dam[mid]', 'dam[in]', 'dam[out]', 'frac_strain', 'frac_strain[mid]', 'frac_strain[in]', 'frac_strain[out]']
        np.testing.assert_equal(svars, expected)

        svars = self.mili.queriable_svars(vector_only=True, show_ips=True)
        expected = ['cpu_time', 'cpu_time[init]', 'cpu_time[plot]', 'cpu_time[hsp]', 'cpu_time[other_i_o]', 'cpu_time[brick]', 'cpu_time[beam]',
                    'cpu_time[shell]', 'cpu_time[tshell]', 'cpu_time[discrete]', 'cpu_time[delam]', 'cpu_time[cohesive]', 'cpu_time[ml]',
                    'cpu_time[ntet]', 'cpu_time[sph]', 'cpu_time[kin_contact]', 'cpu_time[reglag_contact]', 'cpu_time[lag_solver]', 'cpu_time[coupling]',
                    'cpu_time[solution]', 'cpu_time[xfem]', 'cpu_time[total]', 'cseg_var', 'cseg_var[sn]', 'cseg_var[shmag]', 'cseg_var[sr]',
                    'cseg_var[ss]', 'cseg_var[s1]', 'cseg_var[s2]', 'cseg_var[s3]', 'nodpos', 'nodpos[ux]', 'nodpos[uy]', 'nodpos[uz]', 'nodvel',
                    'nodvel[vx]', 'nodvel[vy]', 'nodvel[vz]', 'nodacc', 'nodacc[ax]', 'nodacc[ay]', 'nodacc[az]', 'stress', 'stress[sx]', 'stress[sy]',
                    'stress[sz]', 'stress[sxy]', 'stress[syz]', 'stress[szx]', 'es_1a[0-4][sx]', 'es_1a[0-4][sy]', 'es_1a[0-4][sz]', 'es_1a[0-4][sxy]',
                    'es_1a[0-4][syz]', 'es_1a[0-4][szx]', 'es_1a[0-4][eps]', 'strain', 'strain[ex]', 'strain[ey]', 'strain[ez]', 'strain[exy]', 'strain[eyz]',
                    'strain[ezx]', 'es_3a[0-2][sx]', 'es_3a[0-2][sy]', 'es_3a[0-2][sz]', 'es_3a[0-2][sxy]', 'es_3a[0-2][syz]', 'es_3a[0-2][szx]',
                    'es_3c[0-2][ex]', 'es_3c[0-2][ey]', 'es_3c[0-2][ez]', 'es_3c[0-2][exy]', 'es_3c[0-2][eyz]', 'es_3c[0-2][ezx]', 'es_3c[0-2][edrate]',
                    'svec', 'svec[svec_x]', 'svec[svec_y]', 'svec[svec_z]', 'stress_mid', 'stress_mid[sx]', 'stress_mid[sy]', 'stress_mid[sz]',
                    'stress_mid[sxy]', 'stress_mid[syz]', 'stress_mid[szx]', 'stress_in', 'stress_in[sx]', 'stress_in[sy]', 'stress_in[sz]', 'stress_in[sxy]',
                    'stress_in[syz]', 'stress_in[szx]', 'stress_out', 'stress_out[sx]', 'stress_out[sy]', 'stress_out[sz]', 'stress_out[sxy]',
                    'stress_out[syz]', 'stress_out[szx]', 'bend', 'bend[mxx]', 'bend[myy]', 'bend[mxy]', 'shear', 'shear[qxx]', 'shear[qyy]', 'normal',
                    'normal[nxx]', 'normal[nyy]', 'normal[nxy]', 'press_cut', 'press_cut[mid]', 'press_cut[in]', 'press_cut[out]', 'd_1', 'd_1[mid]',
                    'd_1[in]', 'd_1[out]', 'd_2', 'd_2[mid]', 'd_2[in]', 'd_2[out]', 'dam', 'dam[mid]', 'dam[in]', 'dam[out]', 'frac_strain',
                    'frac_strain[mid]', 'frac_strain[in]', 'frac_strain[out]']
        np.testing.assert_equal(svars, expected)

    #==============================================================================
    def test_supported_variables(self):
        EXPECTED = ['disp_x', 'disp_y', 'disp_z', 'disp_mag', 'disp_rad_mag_xy',
                    'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z', 'vol_strain',
                    'prin_strain1', 'prin_strain2', 'prin_strain3',
                    'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3',
                    'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt',
                    'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt',
                    'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure',
                    'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3', 'max_shear_stress',
                    'triaxiality', 'eps_rate', 'nodtangmag', 'mat_cog_disp_x', 'mat_cog_disp_y',
                    'mat_cog_disp_z', 'element_volume',
                    ]
        supported_variables = self.mili.supported_derived_variables()
        self.assertEqual( EXPECTED, supported_variables )

    #==============================================================================
    def test_derived_variables_of_class(self):
        BRICK_DERIVED = [
            "vol_strain", "prin_strain1", "prin_strain2", "prin_strain3", "prin_dev_strain1",
            "prin_dev_strain2", "prin_dev_strain3", "prin_strain1_alt", "prin_strain2_alt",
            "prin_strain3_alt", "prin_dev_strain1_alt", "prin_dev_strain2_alt", "prin_dev_strain3_alt",
            "prin_stress1", "prin_stress2", "prin_stress3", "eff_stress", "pressure", "prin_dev_stress1",
            "prin_dev_stress2", "prin_dev_stress3", "max_shear_stress", "triaxiality", 'element_volume'
        ]
        BEAM_DERIVED = [
            "prin_stress1", "prin_stress2", "prin_stress3", "eff_stress", "pressure", "prin_dev_stress1",
            "prin_dev_stress2", "prin_dev_stress3", "max_shear_stress", "triaxiality", "eps_rate"
        ]
        SHELL_DERIVED = [
            "vol_strain", "prin_strain1", "prin_strain2", "prin_strain3", "prin_dev_strain1",
            "prin_dev_strain2", "prin_dev_strain3", "prin_strain1_alt", "prin_strain2_alt",
            "prin_strain3_alt", "prin_dev_strain1_alt", "prin_dev_strain2_alt", "prin_dev_strain3_alt",
            "prin_stress1", "prin_stress2", "prin_stress3", "eff_stress", "pressure", "prin_dev_stress1",
            "prin_dev_stress2", "prin_dev_stress3", "max_shear_stress", "triaxiality"
        ]
        CSEG_DERIVED = []
        NODE_DERIVED = [
            "disp_x", "disp_y", "disp_z", "disp_mag", "disp_rad_mag_xy", "vel_x", "vel_y", "vel_z",
            "acc_x", "acc_y", "acc_z"
        ]
        self.assertEqual( self.mili.derived_variables_of_class("brick"), BRICK_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("beam"), BEAM_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("shell"), SHELL_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("cseg"), CSEG_DERIVED )
        self.assertEqual( self.mili.derived_variables_of_class("node"), NODE_DERIVED )

    #==============================================================================
    def test_classes_of_derived_variable(self):
        DISPX_CLASSES = ["node"]
        EFF_STRESS_CLASSES = ["beam", "brick", "shell"]
        VOL_STRAIN_CLASSES = ["brick", "shell"]
        VOLUME_CLASSES = ["brick"]
        self.assertEqual( self.mili.classes_of_derived_variable("disp_x"), DISPX_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("eff_stress"), EFF_STRESS_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("vol_strain"), VOL_STRAIN_CLASSES)
        self.assertEqual( self.mili.classes_of_derived_variable("element_volume"), VOLUME_CLASSES)

    #==============================================================================
    def test_labels(self):
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

        labels = self.mili.labels("node")
        np.testing.assert_equal(labels, NODE_LBLS)

        labels = self.mili.labels("brick")
        np.testing.assert_equal(labels, BRICK_LBLS)

    #==============================================================================
    def test_materials(self):
        mats = self.mili.materials()
        self.assertEqual(mats, {'es_1': [1], 'es_12': [2], 'es_13': [3], 'slide1s': [4], 'slide1m': [5]})

    #==============================================================================
    def test_material_numbers(self):
        mat_nums = self.mili.material_numbers()
        np.testing.assert_equal(mat_nums, [1,2,3,4,5])

    #==============================================================================
    def test_connectivity(self):
        all_conn = self.mili.connectivity()

        expected_conn = {
            'beam': np.array([[ 1,  3,  2,  1],
                              [ 3,  5,  4,  1],
                              [ 5,  7,  6,  1],
                              [12, 13,  2,  1],
                              [13, 14,  4,  1],
                              [14, 15,  6,  1],
                              [ 1, 17, 16,  1],
                              [17, 18, 16,  1],
                              [18, 19, 16,  1],
                              [19, 20, 16,  1],
                              [20,  8, 16,  1],
                              [ 3, 21, 16,  1],
                              [21, 22, 16,  1],
                              [22, 23, 16,  1],
                              [23, 24, 16,  1],
                              [24,  9, 16,  1],
                              [ 5, 25, 16,  1],
                              [25, 26, 16,  1],
                              [26, 27, 16,  1],
                              [27, 28, 16,  1],
                              [28, 10, 16,  1],
                              [ 7, 29, 16,  1],
                              [29, 30, 16,  1],
                              [30, 31, 16,  1],
                              [31, 32, 16,  1],
                              [32, 11, 16,  1],
                              [12, 33, 16,  1],
                              [33, 34, 16,  1],
                              [34, 35, 16,  1],
                              [35, 36, 16,  1],
                              [36,  8, 16,  1],
                              [13, 37, 16,  1],
                              [37, 38, 16,  1],
                              [38, 39, 16,  1],
                              [39, 40, 16,  1],
                              [40,  9, 16,  1],
                              [14, 41, 16,  1],
                              [41, 42, 16,  1],
                              [42, 43, 16,  1],
                              [43, 44, 16,  1],
                              [44, 10, 16,  1],
                              [15, 45, 16,  1],
                              [45, 46, 16,  1],
                              [46, 47, 16,  1],
                              [47, 48, 16,  1],
                              [48, 11, 16,  1]], dtype=np.int32),
            'brick': np.array([[ 65,  81,  85,  69,  66,  82,  86,  70,   2],
                               [ 81,  97, 101,  85,  82,  98, 102,  86,   2],
                               [ 69,  85,  89,  73,  70,  86,  90,  74,   2],
                               [ 85, 101, 105,  89,  86, 102, 106,  90,   2],
                               [ 73,  89,  93,  77,  74,  90,  94,  78,   2],
                               [ 89, 105, 109,  93,  90, 106, 110,  94,   2],
                               [ 66,  82,  86,  70,  67,  83,  87,  71,   2],
                               [ 82,  98, 102,  86,  83,  99, 103,  87,   2],
                               [ 70,  86,  90,  74,  71,  87,  91,  75,   2],
                               [ 86, 102, 106,  90,  87, 103, 107,  91,   2],
                               [ 74,  90,  94,  78,  75,  91,  95,  79,   2],
                               [ 90, 106, 110,  94,  91, 107, 111,  95,   2],
                               [ 67,  83,  87,  71,  68,  84,  88,  72,   2],
                               [ 83,  99, 103,  87,  84, 100, 104,  88,   2],
                               [ 71,  87,  91,  75,  72,  88,  92,  76,   2],
                               [ 87, 103, 107,  91,  88, 104, 108,  92,   2],
                               [ 75,  91,  95,  79,  76,  92,  96,  80,   2],
                               [ 91, 107, 111,  95,  92, 108, 112,  96,   2],
                               [ 97, 113, 117, 101,  98, 114, 118, 102,   2],
                               [113, 129, 133, 117, 114, 130, 134, 118,   2],
                               [101, 117, 121, 105, 102, 118, 122, 106,   2],
                               [117, 133, 137, 121, 118, 134, 138, 122,   2],
                               [105, 121, 125, 109, 106, 122, 126, 110,   2],
                               [121, 137, 141, 125, 122, 138, 142, 126,   2],
                               [ 98, 114, 118, 102,  99, 115, 119, 103,   2],
                               [114, 130, 134, 118, 115, 131, 135, 119,   2],
                               [102, 118, 122, 106, 103, 119, 123, 107,   2],
                               [118, 134, 138, 122, 119, 135, 139, 123,   2],
                               [106, 122, 126, 110, 107, 123, 127, 111,   2],
                               [122, 138, 142, 126, 123, 139, 143, 127,   2],
                               [ 99, 115, 119, 103, 100, 116, 120, 104,   2],
                               [115, 131, 135, 119, 116, 132, 136, 120,   2],
                               [103, 119, 123, 107, 104, 120, 124, 108,   2],
                               [119, 135, 139, 123, 120, 136, 140, 124,   2],
                               [107, 123, 127, 111, 108, 124, 128, 112,   2],
                               [123, 139, 143, 127, 124, 140, 144, 128,   2]], dtype=np.int32),
            'shell': np.array([[49, 53, 54, 50,  3],
                               [53, 12, 13, 54,  3],
                               [50, 54, 55, 51,  3],
                               [54, 13, 14, 55,  3],
                               [51, 55, 56, 52,  3],
                               [55, 14, 15, 56,  3],
                               [12, 57, 58, 13,  3],
                               [57, 61, 62, 58,  3],
                               [13, 58, 59, 14,  3],
                               [58, 62, 63, 59,  3],
                               [14, 59, 60, 15,  3],
                               [59, 63, 64, 60,  3]], dtype=np.int32),
            'cseg': np.array([[ 49,  53,  54,  50,   4],
                              [ 50,  54,  55,  51,   4],
                              [ 51,  55,  56,  52,   4],
                              [ 12,  13,  54,  53,   4],
                              [ 13,  14,  55,  54,   4],
                              [ 14,  15,  56,  55,   4],
                              [ 12,  57,  58,  13,   4],
                              [ 13,  58,  59,  14,   4],
                              [ 14,  59,  60,  15,   4],
                              [ 57,  61,  62,  58,   4],
                              [ 58,  62,  63,  59,   4],
                              [ 59,  63,  64,  60,   4],
                              [ 65,  69,  85,  81,   5],
                              [ 69,  73,  89,  85,   5],
                              [ 73,  77,  93,  89,   5],
                              [ 81,  85, 101,  97,   5],
                              [ 85,  89, 105, 101,   5],
                              [ 89,  93, 109, 105,   5],
                              [ 97, 101, 117, 113,   5],
                              [101, 105, 121, 117,   5],
                              [105, 109, 125, 121,   5],
                              [113, 117, 133, 129,   5],
                              [117, 121, 137, 133,   5],
                              [121, 125, 141, 137,   5]], dtype=np.int32)
        }

        np.testing.assert_equal(all_conn, expected_conn)

        brick_conn = self.mili.connectivity("brick")
        np.testing.assert_equal(brick_conn, expected_conn["brick"])

    #==============================================================================
    def test_material_classes(self):
        mat_classes = self.mili.material_classes(1)
        self.assertEqual(mat_classes, ["beam"])

        mat_classes = self.mili.material_classes(2)
        self.assertEqual(mat_classes, ["brick"])

        mat_classes = self.mili.material_classes(3)
        self.assertEqual(mat_classes, ["shell"])

        mat_classes = self.mili.material_classes(4)
        self.assertEqual(mat_classes, ["cseg"])

        mat_classes = self.mili.material_classes(5)
        self.assertEqual(mat_classes, ["cseg"])

    #==============================================================================
    def test_classes_of_state_variable(self):
        sx_classes = self.mili.classes_of_state_variable('sx')
        self.assertEqual(set(sx_classes), set(["beam", "shell", "brick"]))

        uy_classes = self.mili.classes_of_state_variable('uy')
        self.assertEqual(uy_classes, ["node"])

        axf_classes = self.mili.classes_of_state_variable('axf')
        self.assertEqual(axf_classes, ["beam"])

    #==============================================================================
    def test_state_variables_of_class(self):
        glob_vars = self.mili.state_variables_of_class("glob")
        mat_vars = self.mili.state_variables_of_class("mat")
        beam_vars = self.mili.state_variables_of_class("beam")
        brick_vars = self.mili.state_variables_of_class("brick")

        self.assertEqual(
            glob_vars,
            ['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
             'tcon_fric_eng', 'tcon_eng', 'ew', 'te', 'rbvx', 'rbvy', 'rbvz', 'rbax', 'rbay',
             'rbaz', 'init', 'plot', 'hsp', 'other_i_o', 'brick', 'beam', 'shell', 'tshell',
             'discrete', 'delam', 'cohesive', 'ml', 'ntet', 'sph', 'kin_contact', 'reglag_contact',
             'lag_solver', 'coupling', 'solution', 'xfem', 'total', 'cpu_time']
        )
        self.assertEqual(
            mat_vars,
            ['matpe', 'matke', 'mathe', 'matbve', 'matdre', 'matstde', 'matflde', 'matte',
             'matmass', 'matcgx', 'matcgy', 'matcgz', 'matxv', 'matyv', 'matzv', 'matxa', 'matya',
             'matza', 'con_forx', 'con_fory', 'con_forz', 'con_momx', 'con_momy', 'con_momz',
             'con_damp_eng', 'con_fric_eng', 'con_eng']
        )
        self.assertEqual(
            beam_vars,
            ['sx', 'sy', 'sz', 'sxy', 'syz', 'szx', 'eps', 'es_1a', 'axf', 'sfs', 'sft', 'ms', 'mt',
             'tor', 'svec_x', 'svec_y', 'svec_z', 'svec', 'sand', 'cause']
        )
        self.assertEqual(
            brick_vars,
            ['sx', 'sy', 'sz', 'sxy', 'syz', 'szx', 'stress', 'ex', 'ey', 'ez',
             'exy', 'eyz', 'ezx', 'strain', 'edrate', 'sand', 'cause']
        )

    #==============================================================================
    def test_containing_state_variables_of_class(self):
        containing = self.mili.containing_state_variables_of_class("sx", "brick")
        self.assertEqual(containing, ["stress"])

        containing = self.mili.containing_state_variables_of_class("ux", "node")
        self.assertEqual(containing, ["nodpos"])

        containing = self.mili.containing_state_variables_of_class("ux", "brick")
        self.assertEqual(containing, [])

    #==============================================================================
    def test_components_of_vector_svar(self):
        comps = self.mili.components_of_vector_svar("stress")
        self.assertEqual(comps, ["sx", "sy", "sz", "sxy", "syz", "szx"])

    #==============================================================================
    def test_parts_of_class_name(self):
        brick_parts = self.mili.parts_of_class_name("brick")
        beam_parts = self.mili.parts_of_class_name("beam")
        shell_parts = self.mili.parts_of_class_name("shell")
        cseg_parts = self.mili.parts_of_class_name("cseg")

        self.assertEqual( brick_parts.size, 36 )
        np.testing.assert_equal( np.unique(brick_parts), np.array([2]) )

        self.assertEqual( beam_parts.size, 46 )
        np.testing.assert_equal( np.unique(beam_parts), np.array([1]) )

        self.assertEqual( shell_parts.size, 12 )
        np.testing.assert_equal( np.unique(shell_parts), np.array([3]) )

        self.assertEqual( cseg_parts.size, 24 )
        np.testing.assert_equal( np.unique(cseg_parts), np.array([1]) )

    #==============================================================================
    def test_materials_of_class_name(self):
        brick_mats = self.mili.materials_of_class_name("brick")
        beam_mats = self.mili.materials_of_class_name("beam")
        shell_mats = self.mili.materials_of_class_name("shell")
        cseg_mats = self.mili.materials_of_class_name("cseg")

        self.assertEqual( brick_mats.size, 36 )
        np.testing.assert_equal( np.unique(brick_mats), np.array([2]) )

        self.assertEqual( beam_mats.size, 46 )
        np.testing.assert_equal( np.unique(beam_mats), np.array([1]) )

        self.assertEqual( shell_mats.size, 12 )
        np.testing.assert_equal( np.unique(shell_mats), np.array([3]) )

        self.assertEqual( cseg_mats.size, 24 )
        np.testing.assert_equal( cseg_mats, np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
                                                      5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]) )

    #==============================================================================
    def test_class_labels_of_mat(self):
        answer = self.mili.class_labels_of_material(5,'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

        answer = self.mili.class_labels_of_material("5",'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

        answer = self.mili.class_labels_of_material("slide1m",'cseg')
        np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

    #==============================================================================
    def test_all_labels_of_material(self):
       answer = self.mili.all_labels_of_material('es_13')
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

       answer = self.mili.all_labels_of_material('3')
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

       answer = self.mili.all_labels_of_material(3)
       _, labels = list(answer.items())[0]
       self.assertEqual(labels.size, 12)
       np.testing.assert_equal( labels, np.arange( 1, 13, dtype = np.int32 ) )

    #==============================================================================
    def test_nodes_label(self):
        nodes, elem_labels = self.mili.nodes_of_elems('brick', 1)
        self.assertEqual( elem_labels[0], 1 )
        self.assertEqual(nodes.size, 8)
        np.testing.assert_equal(nodes, np.array( [[65, 81, 85, 69, 66, 82, 86, 70]], dtype = np.int32 ))

    #==============================================================================
    def test_nodes_material(self):
        answer = self.mili.nodes_of_material('es_1')
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material('1')
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )

        answer = self.mili.nodes_of_material(1)
        self.assertEqual(answer.size, 48)
        np.testing.assert_equal( answer, np.arange(1, 49, dtype = np.int32) )