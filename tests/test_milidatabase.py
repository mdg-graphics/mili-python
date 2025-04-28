#!/usr/bin/env python3
"""
Testing for the MiliDatabase Module/class.

SPDX-License-Identifier: (MIT)
"""

import os
import unittest
from mili import reader
from mili.milidatabase import MiliPythonError, ResultModifier
from mili.datatypes import Superclass, Metadata
from mili.mdg_defines import EntityType, NodalStateVariables, StressStrainStateVariables, MaterialStateVariables, GlobalStateVariables
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))


class TestReturnCodes(unittest.TestCase):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def test_loopwrapper(self):
        mili = reader.open_database( TestReturnCodes.file_name, suppress_parallel=True, merge_results=False )
        # All procs return ReturnCode.ERROR so Exception is raised
        with self.assertRaises(MiliPythonError):
            res = mili.query("does-not-exist", "brick")
        # This should pass because only some processors return errors
        res = mili.query("s1", "cseg", states = [1] )

    def test_serverwrapper(self):
        mili = reader.open_database( TestReturnCodes.file_name, suppress_parallel=False, merge_results=False )
        # All procs return ReturnCode.ERROR so Exception is raised
        with self.assertRaises(MiliPythonError):
            res = mili.query("does-not-exist", "brick")
        # Test some procs returning ReturnCode.ERROR so call succeeds
        res = mili.query("s1", "cseg", states = [1] )
        mili.close()


class SharedSerialTests:
    class SerialTests(unittest.TestCase):
        #==============================================================================
        def test_invalid_inputs(self):
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(4, 'node', labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 3, ips = 'cat')
            with self.assertRaises(TypeError):
                # class_name should be class_sname
                self.mili.query(svar_names="sx", class_name="brick")
            with self.assertRaises(TypeError):
                self.mili.query("sx")

        #==============================================================================
        def test_nodes_getter(self):
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
        def test_statemaps_getter(self):
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            state_maps = self.mili.state_maps()
            self.assertEqual(STATE_COUNT, len(state_maps))
            self.assertEqual(FIRST_STATE, state_maps[0].time)
            self.assertEqual(LAST_STATE, state_maps[-1].time)

        #==============================================================================
        def test_times(self):
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            times = self.mili.times()
            self.assertEqual(STATE_COUNT, len(times))
            self.assertEqual(FIRST_STATE, times[0])
            self.assertEqual(LAST_STATE, times[-1])

            times = self.mili.times([1,101])
            self.assertEqual(FIRST_STATE, times[0])
            self.assertEqual(LAST_STATE, times[-1])

        #==============================================================================
        def test_labels_getter(self):
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

            np.testing.assert_equal(self.mili.labels('node'), NODE_LBLS)
            np.testing.assert_equal(self.mili.labels('beam'), BEAM_LBLS)
            np.testing.assert_equal(self.mili.labels('brick'), BRICK_LBLS)
            np.testing.assert_equal(self.mili.labels('shell'), SHELL_LBLS)
            np.testing.assert_equal(self.mili.labels('glob'), GLOB_LBLS)
            np.testing.assert_equal(self.mili.labels('mat'), MATS_LBLS)

            np.testing.assert_equal(self.mili.labels(EntityType.NODE), NODE_LBLS)
            np.testing.assert_equal(self.mili.labels(EntityType.BEAM), BEAM_LBLS)
            np.testing.assert_equal(self.mili.labels(EntityType.BRICK), BRICK_LBLS)
            np.testing.assert_equal(self.mili.labels(EntityType.SHELL), SHELL_LBLS)
            np.testing.assert_equal(self.mili.labels(EntityType.GLOBAL), GLOB_LBLS)
            np.testing.assert_equal(self.mili.labels(EntityType.MATERIAL), MATS_LBLS)

        #==============================================================================
        def test_connectivity(self):
            all_conn = self.mili.connectivity()
            conn_classes = list(all_conn.keys())
            self.assertEqual(set(conn_classes), set(["beam", "brick", "shell", "cseg"]))
            self.assertEqual(all_conn['beam'].shape, (46,4))
            self.assertEqual(all_conn['brick'].shape, (36,9))
            self.assertEqual(all_conn['shell'].shape, (12,5))
            self.assertEqual(all_conn['cseg'].shape, (24,5))

            self.assertTrue( all( all_conn['beam'][:,-1] == 1) )  # All beams are material 1
            self.assertTrue( all( all_conn['brick'][:,-1] == 2) )  # All bricks are material 2
            self.assertTrue( all( all_conn['shell'][:,-1] == 3) )  # All shells are material 3
            self.assertTrue( all( all_conn['cseg'][1:12,-1] == 4) )  # Csegs 1-12 are material 4
            self.assertTrue( all( all_conn['cseg'][12:24,-1] == 5) )  # Csegs 12-24 are material 5

            self.assertTrue( all( self.mili.connectivity('beam')[:,-1] == 1) )  # All beams are material 1
            self.assertTrue( all( self.mili.connectivity('brick')[:,-1] == 2) )  # All bricks are material 2
            self.assertTrue( all( self.mili.connectivity('shell')[:,-1] == 3) )  # All shells are material 3
            self.assertTrue( all( self.mili.connectivity('cseg')[1:12,-1] == 4) )  # Csegs 1-12 are material 4
            self.assertTrue( all( self.mili.connectivity('cseg')[12:24,-1] == 5) )  # Csegs 12-24 are material 5

            self.assertTrue( all( self.mili.connectivity(EntityType.BEAM)[:,-1] == 1) )  # All beams are material 1
            self.assertTrue( all( self.mili.connectivity(EntityType.BRICK)[:,-1] == 2) )  # All bricks are material 2
            self.assertTrue( all( self.mili.connectivity(EntityType.SHELL)[:,-1] == 3) )  # All shells are material 3
            self.assertTrue( all( self.mili.connectivity(EntityType.CONTACT_SEGMENT)[1:12,-1] == 4) )  # Csegs 1-12 are material 4
            self.assertTrue( all( self.mili.connectivity(EntityType.CONTACT_SEGMENT)[12:24,-1] == 5) )  # Csegs 12-24 are material 5

        #==============================================================================
        def test_faces(self):
            with self.assertRaises(MiliPythonError):
                self.mili.faces("brickkkkk", 1)
            with self.assertRaises(MiliPythonError):
                self.mili.faces("shell", 1)
            with self.assertRaises(MiliPythonError):
                self.mili.faces(EntityType.SHELL, 1)
            with self.assertRaises(MiliPythonError):
                self.mili.faces("brick", 100)

            faces = self.mili.faces("brick", 1)
            np.testing.assert_equal(faces[1], [81, 85, 86, 82])
            np.testing.assert_equal(faces[2], [85, 69, 70, 86])
            np.testing.assert_equal(faces[3], [65, 66, 70, 69])
            np.testing.assert_equal(faces[4], [81, 82, 66, 65])
            np.testing.assert_equal(faces[5], [66, 82, 86, 70])
            np.testing.assert_equal(faces[6], [65, 69, 85, 81])

            faces = self.mili.faces("brick", 20)
            np.testing.assert_equal(faces[1], [129, 133, 134, 130])
            np.testing.assert_equal(faces[2], [133, 117, 118, 134])
            np.testing.assert_equal(faces[3], [113, 114, 118, 117])
            np.testing.assert_equal(faces[4], [129, 130, 114, 113])
            np.testing.assert_equal(faces[5], [114, 130, 134, 118])
            np.testing.assert_equal(faces[6], [113, 117, 133, 129])

            faces = self.mili.faces(EntityType.BRICK, 32)
            np.testing.assert_equal(faces[1], [131, 135, 136, 132])
            np.testing.assert_equal(faces[2], [135, 119, 120, 136])
            np.testing.assert_equal(faces[3], [115, 116, 120, 119])
            np.testing.assert_equal(faces[4], [131, 132, 116, 115])
            np.testing.assert_equal(faces[5], [116, 132, 136, 120])
            np.testing.assert_equal(faces[6], [115, 119, 135, 131])

        #==============================================================================
        def test_components_of_vector_svar(self):
            comps = self.mili.components_of_vector_svar(StressStrainStateVariables.STRESS)
            self.assertEqual(comps, ["sx", "sy", "sz", "sxy", "syz", "szx"])

            with self.assertRaises(MiliPythonError):
                comps = self.mili.components_of_vector_svar("Does-not-exists")

            with self.assertRaises(MiliPythonError):
                comps = self.mili.components_of_vector_svar("ux")

        #==============================================================================
        def test_reload_state_maps(self):
            self.mili.reload_state_maps()
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            state_maps = self.mili.state_maps()
            self.assertEqual(STATE_COUNT, len(state_maps))
            self.assertEqual(FIRST_STATE, state_maps[0].time)
            self.assertEqual(LAST_STATE, state_maps[-1].time)

        #==============================================================================
        def test_material_numbers(self):
            mat_nums = self.mili.material_numbers()
            self.assertEqual(set(mat_nums), set([1,2,3,4,5]))

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

            with self.assertRaises(MiliPythonError):
                mat_classes = self.mili.material_classes(None)
                print(mat_classes)

        #==============================================================================
        def test_int_points_of_state_variable(self):
            ipts = self.mili.int_points_of_state_variable("sx", "brick")
            np.testing.assert_equal(ipts, [])

            ipts = self.mili.int_points_of_state_variable("sx", "beam")
            np.testing.assert_equal(ipts, [1,2,3,4])

            ipts = self.mili.int_points_of_state_variable("sx", "shell")
            np.testing.assert_equal(ipts, [1,2])

            ipts = self.mili.int_points_of_state_variable("sx", EntityType.BRICK)
            np.testing.assert_equal(ipts, [])

            ipts = self.mili.int_points_of_state_variable("sx", EntityType.BEAM)
            np.testing.assert_equal(ipts, [1,2,3,4])

            ipts = self.mili.int_points_of_state_variable("sx", EntityType.SHELL)
            np.testing.assert_equal(ipts, [1,2])

        #==============================================================================
        def test_element_sets(self):
            elem_sets = self.mili.element_sets()
            self.assertEqual( set(elem_sets.keys()), set(["es_1", "es_3"]))
            self.assertEqual( elem_sets['es_1'], [1,2,3,4,4] )
            self.assertEqual( elem_sets['es_3'], [1,2,2] )

        #==============================================================================
        def test_integration_points(self):
            int_points = self.mili.integration_points()
            self.assertEqual( set(int_points.keys()), set(["1", "3"]))
            self.assertEqual( int_points['1'], [1,2,3,4] )
            self.assertEqual( int_points['3'], [1,2] )

        #==============================================================================
        def test_derived_variables_of_class(self):
            BRICK_DERIVED = ['vol_strain', 'prin_strain1', 'prin_strain2', 'prin_strain3', 'prin_dev_strain1', 'prin_dev_strain2',
                            'prin_dev_strain3', 'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt', 'prin_dev_strain1_alt',
                            'prin_dev_strain2_alt', 'prin_dev_strain3_alt', 'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress',
                            'pressure', 'prin_dev_stress1', 'prin_dev_stress2', 'prin_dev_stress3', 'max_shear_stress', 'triaxiality',
                            'element_volume', 'surfstrainx', 'surfstrainy', 'surfstrainz', 'surfstrainxy', 'surfstrainyz', 'surfstrainzx',
                            ]
            BEAM_DERIVED = ['prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure', 'prin_dev_stress1', 'prin_dev_stress2',
                            'prin_dev_stress3', 'max_shear_stress', 'triaxiality', 'eps_rate'
                            ]
            SHELL_DERIVED = ['vol_strain', 'prin_strain1', 'prin_strain2', 'prin_strain3', 'prin_dev_strain1', 'prin_dev_strain2', 'prin_dev_strain3',
                            'prin_strain1_alt', 'prin_strain2_alt', 'prin_strain3_alt', 'prin_dev_strain1_alt', 'prin_dev_strain2_alt', 'prin_dev_strain3_alt',
                            'prin_stress1', 'prin_stress2', 'prin_stress3', 'eff_stress', 'pressure', 'prin_dev_stress1', 'prin_dev_stress2',
                            'prin_dev_stress3', 'max_shear_stress', 'triaxiality', 'area'
                            ]
            CSEG_DERIVED = ['area']
            NODE_DERIVED = ['disp_x', 'disp_y', 'disp_z', 'disp_mag', 'disp_rad_mag_xy', 'vel_x', 'vel_y', 'vel_z', 'acc_x', 'acc_y', 'acc_z',]
            self.assertEqual( self.mili.derived_variables_of_class("brick"), BRICK_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class("beam"), BEAM_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class("shell"), SHELL_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class("cseg"), CSEG_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class("node"), NODE_DERIVED )

            self.assertEqual( self.mili.derived_variables_of_class(EntityType.BRICK), BRICK_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class(EntityType.BEAM), BEAM_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class(EntityType.SHELL), SHELL_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class(EntityType.CONTACT_SEGMENT), CSEG_DERIVED )
            self.assertEqual( self.mili.derived_variables_of_class(EntityType.NODE), NODE_DERIVED )

        #==============================================================================
        def test_classes_of_state_variable(self):
            sx_classes = self.mili.classes_of_state_variable('sx')
            self.assertEqual(set(sx_classes), set(["beam", "shell", "brick"]))

            uy_classes = self.mili.classes_of_state_variable(NodalStateVariables.Y_POSITION)
            self.assertEqual(uy_classes, ["node"])

            axf_classes = self.mili.classes_of_state_variable('axf')
            self.assertEqual(axf_classes, ["beam"])

        #==============================================================================
        def test_materials_of_class_name(self):
            brick_mats = self.mili.materials_of_class_name(EntityType.BRICK)
            beam_mats = self.mili.materials_of_class_name("beam")
            shell_mats = self.mili.materials_of_class_name(EntityType.SHELL)
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
        def test_parts_of_class_name(self):
            brick_parts = self.mili.parts_of_class_name("brick")
            beam_parts = self.mili.parts_of_class_name(EntityType.BEAM)
            shell_parts = self.mili.parts_of_class_name("shell")
            cseg_parts = self.mili.parts_of_class_name(EntityType.CONTACT_SEGMENT)

            self.assertEqual( brick_parts.size, 36 )
            np.testing.assert_equal( np.unique(brick_parts), np.array([2]) )

            self.assertEqual( beam_parts.size, 46 )
            np.testing.assert_equal( np.unique(beam_parts), np.array([1]) )

            self.assertEqual( shell_parts.size, 12 )
            np.testing.assert_equal( np.unique(shell_parts), np.array([3]) )

            self.assertEqual( cseg_parts.size, 24 )
            np.testing.assert_equal( np.unique(cseg_parts), np.array([1]) )

        #==============================================================================
        def test_mesh_object_classes_getter(self):
            mo_classes = self.mili._mili.mesh_object_classes()

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

        #==============================================================================
        def test_nodes_of_elems(self):
            nodes, elem_labels = self.mili.nodes_of_elems('brick', 1)
            self.assertEqual( elem_labels[0], 1 )
            self.assertEqual(nodes.size, 8)
            np.testing.assert_equal(nodes, np.array( [[65, 81, 85, 69, 66, 82, 86, 70]], dtype = np.int32 ))

            nodes, elem_labels = self.mili.nodes_of_elems(EntityType.BRICK, 1)
            self.assertEqual( elem_labels[0], 1 )
            self.assertEqual(nodes.size, 8)
            np.testing.assert_equal(nodes, np.array( [[65, 81, 85, 69, 66, 82, 86, 70]], dtype = np.int32 ))

        #==============================================================================
        def test_class_labels_of_mat(self):
            answer = self.mili.class_labels_of_material(5,'cseg')
            np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

            answer = self.mili.class_labels_of_material("5", EntityType.CONTACT_SEGMENT)
            np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))

            answer = self.mili.class_labels_of_material("slide1m",'cseg')
            np.testing.assert_equal(answer, np.array([13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24],dtype=np.int32))


class SerialSingleStateFile(SharedSerialTests.SerialTests):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( SerialSingleStateFile.file_name, suppress_parallel = True, merge_results=False )

    # Tests unique to single state file
    #==============================================================================
    def test_metadata(self):
        metadata = self.mili.metadata()
        EXPECTED = Metadata(
            code_name = "",
            username = "legler5",
            job_id = "",
            nprocs = 1,
            date = "Wed Mar 20 14:26:44 2019",
            host_name = "rzgenie2",
            library_version = "V16_01"
        )
        np.testing.assert_equal(metadata, EXPECTED)

    #==============================================================================
    def test_state_variables(self):
        state_variable_names = set(self.mili._mili.state_variables().keys())
        SVAR_NAMES = set(['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
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
                          'dam', 'frac_strain', 'sand', 'cause'])
        self.assertEqual(state_variable_names, SVAR_NAMES)

    #==============================================================================
    def test_state_variables_of_class(self):
        GLOB_SVARS = ['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
                      'tcon_fric_eng', 'tcon_eng', 'ew', 'te', 'rbvx', 'rbvy', 'rbvz', 'rbax', 'rbay',
                      'rbaz', 'init', 'plot', 'hsp', 'other_i_o', 'brick', 'beam', 'shell', 'tshell',
                      'discrete', 'delam', 'cohesive', 'ml', 'ntet', 'sph', 'kin_contact', 'reglag_contact',
                      'lag_solver', 'coupling', 'solution', 'xfem', 'total', 'cpu_time']
        MAT_SVARS = ['matpe', 'matke', 'mathe', 'matbve', 'matdre', 'matstde', 'matflde', 'matte',
            'matmass', 'matcgx', 'matcgy', 'matcgz', 'matxv', 'matyv', 'matzv', 'matxa', 'matya',
            'matza', 'con_forx', 'con_fory', 'con_forz', 'con_momx', 'con_momy', 'con_momz',
            'con_damp_eng', 'con_fric_eng', 'con_eng']
        BEAM_SVARS = ['sx', 'sy', 'sz', 'sxy', 'syz', 'szx', 'eps', 'es_1a', 'axf', 'sfs', 'sft', 'ms', 'mt',
            'tor', 'svec_x', 'svec_y', 'svec_z', 'svec', 'sand', 'cause']
        BRICK_SVARS = ['sx', 'sy', 'sz', 'sxy', 'syz', 'szx', 'stress', 'ex', 'ey', 'ez',
            'exy', 'eyz', 'ezx', 'strain', 'edrate', 'sand', 'cause']

        glob_vars = self.mili.state_variables_of_class("glob")
        mat_vars = self.mili.state_variables_of_class("mat")
        beam_vars = self.mili.state_variables_of_class("beam")
        brick_vars = self.mili.state_variables_of_class("brick")
        self.assertEqual(glob_vars, GLOB_SVARS)
        self.assertEqual(mat_vars, MAT_SVARS)
        self.assertEqual(beam_vars, BEAM_SVARS)
        self.assertEqual(brick_vars, BRICK_SVARS)

        glob_vars = self.mili.state_variables_of_class(EntityType.GLOBAL)
        mat_vars = self.mili.state_variables_of_class(EntityType.MATERIAL)
        beam_vars = self.mili.state_variables_of_class(EntityType.BEAM)
        brick_vars = self.mili.state_variables_of_class(EntityType.BRICK)
        self.assertEqual(glob_vars, GLOB_SVARS)
        self.assertEqual(mat_vars, MAT_SVARS)
        self.assertEqual(beam_vars, BEAM_SVARS)
        self.assertEqual(brick_vars, BRICK_SVARS)

    #==============================================================================
    def test_containing_state_variables_of_class(self):
        containing = self.mili.containing_state_variables_of_class("sx", "brick")
        self.assertEqual(containing, ["stress"])

        containing = self.mili.containing_state_variables_of_class(NodalStateVariables.X_POSITION, "node")
        self.assertEqual(containing, ["nodpos"])

        containing = self.mili.containing_state_variables_of_class("ux", EntityType.BRICK)
        self.assertEqual(containing, [])

    #==============================================================================
    def test_state_variable(self):
        answer = self.mili.query(MaterialStateVariables.CENTER_OF_GRAVITY_X_POSITION, 'mat', labels = [1,2], states = 3 )
        self.assertEqual(list(answer.keys()), ['matcgx'] )
        self.assertEqual(answer['matcgx']['title'], "C.G. X-Position")
        self.assertEqual(answer['matcgx']['class_name'], "mat")
        self.assertEqual(answer['matcgx']['layout']['states'][0], 3)
        self.assertEqual(answer['matcgx']['layout']['components'], ["matcgx"])
        np.testing.assert_allclose(answer['matcgx']['layout']['times'], [2.0e-05])
        np.testing.assert_equal( answer['matcgx']['layout']['labels'], np.array( [ 1, 2 ], dtype = np.int32) )
        np.testing.assert_equal( answer['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32) )

        answer = self.mili.query('matcgx', EntityType.MATERIAL, labels = [1,2], states = -99 )
        self.assertEqual(list(answer.keys()), ['matcgx'] )
        self.assertEqual(answer['matcgx']['title'], "C.G. X-Position")
        self.assertEqual(answer['matcgx']['class_name'], "mat")
        self.assertEqual(answer['matcgx']['layout']['states'][0], 3)
        self.assertEqual(answer['matcgx']['layout']['components'], ["matcgx"])
        np.testing.assert_allclose(answer['matcgx']['layout']['times'], [2.0e-05])
        np.testing.assert_equal( answer['matcgx']['layout']['labels'], np.array( [ 1, 2 ], dtype = np.int32) )
        np.testing.assert_equal( answer['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32) )

    #==============================================================================
    def test_node_attributes(self):
        answer = self.mili.query('nodpos[ux]', 'node', labels = 70, states = 3 )

        self.assertEqual(answer['nodpos[ux]']['title'], "Node Position")
        self.assertEqual(answer['nodpos[ux]']['class_name'], "node")
        self.assertEqual(answer['nodpos[ux]']['layout']['components'], ['ux'])
        self.assertEqual(answer['nodpos[ux]']['layout']['states'], [3])
        np.testing.assert_allclose(answer['nodpos[ux]']['layout']['times'], [2.0e-05])
        self.assertEqual(answer['nodpos[ux]']['layout']['labels'], [70])
        self.assertEqual(answer['nodpos[ux]']['data'][0], 0.4330127537250519 )

        answer = self.mili.query('nodpos[ux]', EntityType.NODE, labels = 70, states = -99 )
        self.assertEqual(answer['nodpos[ux]']['layout']['states'], [3])
        np.testing.assert_allclose(answer['nodpos[ux]']['layout']['times'], [2.0e-05])
        self.assertEqual(answer['nodpos[ux]']['layout']['labels'][0], 70)
        self.assertEqual(answer['nodpos[ux]']['data'][0], 0.4330127537250519 )

        answer = self.mili.query(NodalStateVariables.X_POSITION, EntityType.NODE, labels = 70, states = 3 )
        self.assertEqual(answer['ux']['title'], "X Position")
        self.assertEqual(answer['ux']['class_name'], "node")
        self.assertEqual(answer['ux']['layout']['components'], ['ux'])
        self.assertEqual(answer['ux']['layout']['states'], [3])
        np.testing.assert_allclose(answer['ux']['layout']['times'], [2.0e-05])
        self.assertEqual(answer['ux']['layout']['labels'], [70])
        self.assertEqual(answer['ux']['data'][0], 0.4330127537250519)

        answer = self.mili.query('ux', 'node', labels = 70, states = -99 )
        self.assertEqual(answer['ux']['layout']['states'], [3])
        np.testing.assert_allclose(answer['ux']['layout']['times'], [2.0e-05])
        self.assertEqual(answer['ux']['layout']['labels'][0], 70)
        self.assertEqual(answer['ux']['data'][0], 0.4330127537250519)

    #==============================================================================
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        self.assertEqual(answer['sx']['title'], 'X Stress')
        self.assertEqual(answer['sx']['class_name'], 'brick')
        self.assertEqual(answer['sx']['layout']['components'], ['sx'])
        self.assertEqual(answer['sx']['layout']['states'], [37])
        np.testing.assert_allclose(answer['sx']['layout']['times'], [3.6e-04])
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

        answer = self.mili.query('sx', 'brick', material = 2, states = -65 )
        self.assertEqual(answer['sx']['layout']['states'], [37])
        np.testing.assert_allclose(answer['sx']['layout']['times'], [3.6e-04])
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

        answer = self.mili.query('sx', EntityType.BRICK, material = 'es_12', states = 37 )
        self.assertEqual(answer['sx']['title'], 'X Stress')
        self.assertEqual(answer['sx']['class_name'], 'brick')
        self.assertEqual(answer['sx']['layout']['components'], ['sx'])
        self.assertEqual(answer['sx']['layout']['states'], [37])
        np.testing.assert_allclose(answer['sx']['layout']['times'], [3.6e-04])
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

        answer = self.mili.query(StressStrainStateVariables.X_STRESS, EntityType.BRICK, material = 'es_12', states = -65 )
        self.assertEqual(answer['sx']['layout']['states'], [37])
        np.testing.assert_allclose(answer['sx']['layout']['times'], [3.6e-04])
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

    #==============================================================================
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        self.assertEqual( answer['nodpos']['title'], 'Node Position')
        self.assertEqual( answer['nodpos']['class_name'], 'node')
        self.assertEqual( answer['nodpos']['layout']['components'], ['ux', 'uy', 'uz'] )
        np.testing.assert_equal(answer['nodpos']['layout']['labels'], [70])
        np.testing.assert_equal(answer['nodpos']['layout']['states'], [4])
        np.testing.assert_allclose(answer['nodpos']['layout']['times'], [3.0e-05])
        np.testing.assert_equal( answer['nodpos']['data'][0,:,:], np.array( [ [ 0.4330127537250519, 0.2500000596046448, 2.436666965484619 ] ], dtype = np.float32 ) )

    #==============================================================================
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        self.assertEqual( answer['stress']['title'], 'Stress')
        self.assertEqual( answer['stress']['class_name'], 'beam')
        np.testing.assert_equal(answer['stress']['layout']['labels'], [5])
        np.testing.assert_equal(answer['stress']['layout']['states'], [21,22])
        np.testing.assert_allclose(answer['stress']['layout']['times'], [2.0e-04, 2.1e-04])
        self.assertEqual( answer['stress']['layout']['components'], ['sx ipt. 2', 'sy ipt. 2', 'sz ipt. 2', 'sxy ipt. 2', 'syz ipt. 2', 'szx ipt. 2'] )
        np.testing.assert_equal( answer['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307 ] ], dtype = np.float32 ) )

    #==============================================================================
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        self.assertEqual( answer['stress[sy]']['title'], 'Stress')
        self.assertEqual( answer['stress[sy]']['class_name'], 'beam')
        self.assertEqual( answer['stress[sy]']['layout']['components'], ['sy ipt. 2'] )
        np.testing.assert_equal(answer['stress[sy]']['layout']['labels'], [5])
        np.testing.assert_equal(answer['stress[sy]']['layout']['states'], [71])
        np.testing.assert_allclose(answer['stress[sy]']['layout']['times'], [7.0e-04])
        self.assertEqual(answer['stress[sy]']['data'][0,0,0], -5545.70751953125)

    #==============================================================================
    def test_query_glob_results(self):
        """
        Test querying for results for M_MESH ("glob") element class.
        """
        answer = self.mili.query(GlobalStateVariables.HOURGLASS_ENERGY, "glob", states=[22])
        self.assertEqual( answer['he']['title'], 'Hourglass Energy')
        self.assertEqual( answer['he']['class_name'], 'glob')
        self.assertEqual( answer['he']['layout']['components'], ['he'] )
        np.testing.assert_equal( answer['he']['layout']['labels'], [1] )
        np.testing.assert_equal( answer['he']['layout']['states'], [22] )
        np.testing.assert_allclose( answer['he']['layout']['times'], [2.1e-04] )
        self.assertAlmostEqual( answer["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

        answer = self.mili.query("bve", "glob", states=[22])
        self.assertEqual( answer['bve']['title'], 'Bulk Vis. Energy')
        self.assertEqual( answer['bve']['class_name'], 'glob')
        self.assertEqual( answer['bve']['layout']['components'], ['bve'] )
        np.testing.assert_equal( answer['bve']['layout']['labels'], [1] )
        np.testing.assert_equal( answer['bve']['layout']['states'], [22] )
        np.testing.assert_allclose( answer['bve']['layout']['times'], [2.1e-04] )
        self.assertAlmostEqual( answer["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

        answer = self.mili.query("te", "glob", states=[22])
        self.assertEqual( answer['te']['title'], 'Total Energy')
        self.assertEqual( answer['te']['class_name'], 'glob')
        self.assertEqual( answer['te']['layout']['components'], ['te'] )
        np.testing.assert_equal( answer['te']['layout']['labels'], [1] )
        np.testing.assert_equal( answer['te']['layout']['states'], [22] )
        np.testing.assert_allclose( answer['te']['layout']['times'], [2.1e-04] )
        self.assertAlmostEqual( answer["te"]["data"][0,0,0], 1629.718, delta=1e-4)

    def test_cummin(self):
        """Test cumulative min result modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", labels=[1,7,11], ips=[1], modifier=ResultModifier.CUMMIN)
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "beam")
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "cummin")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,7,11])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

        # Beam 1 has max of 0.0 across all states
        np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

        # Beam 7
        np.testing.assert_allclose(result[svar_name]['data'][0:21,1,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][21,1,:], -2.864564, rtol=2.0e-07)
        np.testing.assert_allclose(result[svar_name]['data'][22,1,:], -4.125320)
        np.testing.assert_allclose(result[svar_name]['data'][48:60,1,:], -3.764672e+02)
        np.testing.assert_allclose(result[svar_name]['data'][63:65,1,:], -6.132592e+02)
        np.testing.assert_allclose(result[svar_name]['data'][65:82,1,:], -6.723501e+02)
        np.testing.assert_allclose(result[svar_name]['data'][82:,1,:], -7.151539e+02)

        # Beam 11
        np.testing.assert_allclose(result[svar_name]['data'][:,2,:], 0.0)

        svar_name = "disp_x"
        result = self.mili.query(svar_name, "node", labels=[1,44], modifier=ResultModifier.CUMMIN)
        self.assertEqual(result[svar_name]["title"], "X Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "cummin")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_x"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,44])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

        # Node 1
        np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)
        # Node 44
        np.testing.assert_allclose(result[svar_name]['data'][:,1,:], 0.0)

    def test_cummax(self):
        """Test cumulative max derived variable."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", labels=[1,7,11], ips=[1], modifier=ResultModifier.CUMMAX)
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "beam")
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "cummax")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,7,11])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

        # Beam 1 has max of 0.0 across all states
        np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

        # Beam 7
        np.testing.assert_allclose(result[svar_name]['data'][0:35,1,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][35:,1,:], 6.722762)

        # Beam 11
        np.testing.assert_allclose(result[svar_name]['data'][0:21,2,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][21,2,:], 0.12996, rtol=4.0e-06)
        np.testing.assert_allclose(result[svar_name]['data'][22,2,:], 5.531850)
        np.testing.assert_allclose(result[svar_name]['data'][52,2,:], 2078.3381)
        np.testing.assert_allclose(result[svar_name]['data'][53:,2,:], 2199.8977)

        svar_name = "disp_x"
        result = self.mili.query(svar_name, "node", labels=[1,44], modifier=ResultModifier.CUMMAX)
        self.assertEqual(result[svar_name]["title"], "X Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "cummax")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_x"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,44])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

        # Node 1 has max of 0.0 across all states
        np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

        # Node 44
        np.testing.assert_allclose(result[svar_name]['data'][0:21,1,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][21,1,:], 4.011393e-05, rtol=2.0e-07)
        np.testing.assert_allclose(result[svar_name]['data'][22,1,:], 2.411008e-04)
        np.testing.assert_allclose(result[svar_name]['data'][57,1,:], 6.289059e-02)
        np.testing.assert_allclose(result[svar_name]['data'][58:,1,:], 6.300801e-02)

    def test_query_min(self):
        """Test min query modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.MIN)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "min")
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "beam")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,6,6])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -1370.853882)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -840.760498)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.MIN)
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "min")
        self.assertEqual(result[svar_name]["title"], "Y Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,1,1])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.0)

    def test_query_min_dataframe(self):
        """Test min query modifier with as_dataframe=True."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.MIN)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["min", "label"])
        np.testing.assert_equal(list(df.index), [21,22,23])

        np.testing.assert_equal(df["label"][21], 1)
        np.testing.assert_equal(df["label"][22], 6)
        np.testing.assert_equal(df["label"][23], 6)
        np.testing.assert_allclose(df["min"][21], 0.0)
        np.testing.assert_allclose(df["min"][22], -1370.853882)
        np.testing.assert_allclose(df["min"][23], -840.760498)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.MIN)
        df = result[svar_name]
        self.assertEqual(list(df.columns), ["min", "label"])
        self.assertEqual(list(df.index), [97,98,99])

        np.testing.assert_equal(df["label"][97], 1)
        np.testing.assert_equal(df["label"][98], 1)
        np.testing.assert_equal(df["label"][99], 1)
        np.testing.assert_allclose(df["min"][97], 0.0)
        np.testing.assert_allclose(df["min"][98], 0.0)
        np.testing.assert_allclose(df["min"][99], 0.0)

    def test_query_max(self):
        """Test max query modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.MAX)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "max")
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "beam")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,5,4])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 307.012909)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 41.389294)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.MAX)
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "max")
        self.assertEqual(result[svar_name]["title"], "Y Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [11,11,11])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.146183, rtol=4.0e-06)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.145862, rtol=10.0e-07)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.145216, rtol=10.0e-07)

    def test_query_max_dataframe(self):
        """Test max query modifier with as_dataframe=True."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.MAX)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["max", "label"])
        np.testing.assert_equal(list(df.index), [21,22,23])

        np.testing.assert_equal(df["label"][21], 1)
        np.testing.assert_equal(df["label"][22], 5)
        np.testing.assert_equal(df["label"][23], 4)
        np.testing.assert_allclose(df["max"][21], 0.0)
        np.testing.assert_allclose(df["max"][22], 307.012909)
        np.testing.assert_allclose(df["max"][23], 41.389294)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.MAX)
        df = result[svar_name]
        self.assertEqual(list(df.columns), ["max", "label"])
        self.assertEqual(list(df.index), [97,98,99])

        np.testing.assert_equal(df["label"][97], 11)
        np.testing.assert_equal(df["label"][98], 11)
        np.testing.assert_equal(df["label"][99], 11)
        np.testing.assert_allclose(df["max"][97], 0.146183, rtol=4.0e-06)
        np.testing.assert_allclose(df["max"][98], 0.145862, rtol=10.0e-07)
        np.testing.assert_allclose(df["max"][99], 0.145216, rtol=10.0e-07)

    def test_query_average(self):
        """Test average query modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.AVERAGE)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "average")
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "beam")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -25.27041)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -23.06732)

        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[-81,-80,-79], ips=[1], modifier=ResultModifier.AVERAGE)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "average")
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -25.27041)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -23.06732)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.AVERAGE)
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "average")
        self.assertEqual(result[svar_name]["title"], "Y Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.010688, rtol=3.0e-05)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.010649, rtol=5.0e-05)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.010595, rtol=5.0e-05)

    def test_query_average_dataframe(self):
        """Test average query modifier with as_dataframe=True."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.AVERAGE)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["average"])
        np.testing.assert_equal(list(df.index), [21,22,23])
        np.testing.assert_allclose(df["average"][21], 0.0)
        np.testing.assert_allclose(df["average"][22], -25.27041)
        np.testing.assert_allclose(df["average"][23], -23.06732)

        svar_name = "sx"
        result = self.mili.query(svar_name, "beam", states=[-81,-80,-79], ips=[1], as_dataframe=True, modifier=ResultModifier.AVERAGE)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["average"])
        np.testing.assert_equal(list(df.index), [21,22,23])
        np.testing.assert_allclose(df["average"][21], 0.0)
        np.testing.assert_allclose(df["average"][22], -25.27041)
        np.testing.assert_allclose(df["average"][23], -23.06732)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.AVERAGE)
        df = result[svar_name]
        self.assertEqual(list(df.columns), ["average"])
        self.assertEqual(list(df.index), [97,98,99])
        np.testing.assert_allclose(df["average"][97], 0.010688, rtol=3.0e-05)
        np.testing.assert_allclose(df["average"][98], 0.010649, rtol=5.0e-05)
        np.testing.assert_allclose(df["average"][99], 0.010595, rtol=5.0e-05)

    def test_query_median(self):
        """Test median query modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], ips=[1], modifier=ResultModifier.MEDIAN)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "median")
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "brick")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,2,3,4,5])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], -1.924881e-09, rtol=1.2e-07)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -7.198997e+03)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -5.880557e+03)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], modifier=ResultModifier.MEDIAN)
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "median")
        self.assertEqual(result[svar_name]["title"], "Y Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.038422, rtol=6.0e-06)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.038286, rtol=9.0e-06)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.038071, rtol=5.0e-06)

    def test_query_median_dataframe(self):
        """Test median query modifier with as_dataframe=True."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], as_dataframe=True, modifier=ResultModifier.MEDIAN)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["median"])
        np.testing.assert_equal(list(df.index), [21,22,23])
        np.testing.assert_allclose(df["median"][21], -1.924881e-09, rtol=1.2e-07)
        np.testing.assert_allclose(df["median"][22], -7.198997e+03)
        np.testing.assert_allclose(df["median"][23], -5.880557e+03)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], as_dataframe=True, modifier=ResultModifier.MEDIAN)
        df = result[svar_name]
        self.assertEqual(list(df.columns), ["median"])
        self.assertEqual(list(df.index), [97,98,99])
        np.testing.assert_allclose(df["median"][97], 0.038422, rtol=6.0e-06)
        np.testing.assert_allclose(df["median"][98], 0.038286, rtol=9.0e-06)
        np.testing.assert_allclose(df["median"][99], 0.038071, rtol=5.0e-06)

    def test_query_stddev(self):
        """Test stddev query modifier."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], ips=[1], modifier=ResultModifier.STDDEV)
        self.assertEqual(result[svar_name]["source"], "primal")
        self.assertEqual(result[svar_name]["modifier"], "stddev")
        self.assertEqual(result[svar_name]["title"], "X Stress")
        self.assertEqual(result[svar_name]["class_name"], "brick")
        self.assertEqual(result[svar_name]["layout"]["components"], ["sx"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
        np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,2,3,4,5])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 7.788077e-09, rtol=1.2e-07)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 9.055870e+03)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 2.214029e+03, rtol=1.2e-07)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], modifier=ResultModifier.STDDEV)
        self.assertEqual(result[svar_name]["source"], "derived")
        self.assertEqual(result[svar_name]["modifier"], "stddev")
        self.assertEqual(result[svar_name]["title"], "Y Displacement")
        self.assertEqual(result[svar_name]["class_name"], "node")
        self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
        np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
        np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

        np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.025586, rtol=1.6e-05)
        np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.025499, rtol=1.0e-05)
        np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.025357, rtol=3.3e-06)

    def test_query_stddev_dataframe(self):
        """Test stddev query modifier with as_dataframe=True."""
        svar_name = "sx"
        result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], as_dataframe=True, modifier=ResultModifier.STDDEV)
        df = result[svar_name]
        np.testing.assert_equal(list(df.columns), ["stddev"])
        np.testing.assert_equal(list(df.index), [21,22,23])
        np.testing.assert_allclose(df["stddev"][21], 7.788077e-09, rtol=1.2e-07)
        np.testing.assert_allclose(df["stddev"][22], 9.055870e+03)
        np.testing.assert_allclose(df["stddev"][23], 2.214029e+03, rtol=1.2e-07)

        svar_name = "disp_y"
        result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], as_dataframe=True, modifier=ResultModifier.STDDEV)
        df = result[svar_name]
        self.assertEqual(list(df.columns), ["stddev"])
        self.assertEqual(list(df.index), [97,98,99])
        np.testing.assert_allclose(df["stddev"][97], 0.025586, rtol=1.6e-05)
        np.testing.assert_allclose(df["stddev"][98], 0.025499, rtol=1.0e-05)
        np.testing.assert_allclose(df["stddev"][99], 0.025357, rtol=3.3e-06)


class SerialMultiStateFile(SharedSerialTests.SerialTests):
    file_name = os.path.join(dir_path,'data','serial','mstate','d3samp6.plt_c')

    def setUp(self):
        self.mili = reader.open_database( SerialMultiStateFile.file_name, suppress_parallel = True, merge_results=False )

    # Tests unique to multi state file
    #==============================================================================
    def test_state_variables(self):
        state_variable_names = set(self.mili._mili.state_variables().keys())
        SVAR_NAMES = set(['ax', 'axf', 'ay', 'az', 'beam', 'bend', 'brick', 'bve', 'cause', 'cohesive', 'con_damp_eng',
                          'con_eng', 'con_forx', 'con_fory', 'con_forz', 'con_fric_eng', 'con_momx', 'con_momy', 'con_momz',
                          'coupling', 'cpu_time', 'cseg_var', 'cw_cohesive', 'cw_coupling', 'cw_delam', 'cw_hsp', 'cw_main',
                          'cw_ml', 'cw_moment', 'cw_mpcheck', 'cw_other', 'cw_rigid_body', 'cw_slide', 'delam', 'discrete',
                          'dre', 'edrate', 'eps', 'es_1a', 'es_3a', 'es_3c', 'ew', 'ex', 'exy', 'ey', 'eyz', 'ez', 'ezx',
                          'flde', 'he', 'hsp', 'init', 'inteng', 'ke', 'ke_part', 'kin_contact', 'lag_solver', 'matbve',
                          'matcgx', 'matcgy', 'matcgz', 'matdre', 'matflde', 'mathe', 'matke', 'matmass', 'matpe', 'matstde',
                          'matte', 'matxa', 'matxv', 'matya', 'matyv', 'matza', 'matzv', 'ml', 'ms', 'mt', 'mxx', 'mxy', 'myy',
                          'nodacc', 'nodpos', 'nodvel', 'normal', 'ntet', 'nxx', 'nxy', 'nyy', 'other_i_o', 'pe', 'plot', 'qxx',
                          'qyy', 'rbax', 'rbay', 'rbaz', 'rbvx', 'rbvy', 'rbvz', 'reglag_contact', 's1', 's2', 's3', 'sand',
                          'sfs', 'sft', 'shear', 'shell', 'shmag', 'sn', 'solution', 'sph', 'sr', 'ss', 'stde', 'strain',
                          'stress', 'svec', 'svec_x', 'svec_y', 'svec_z', 'sx', 'sxy', 'sy', 'syz', 'sz', 'szx', 'tcon_damp_eng',
                          'tcon_eng', 'tcon_fric_eng', 'te', 'thick', 'tor', 'total', 'tshell', 'ux', 'uy', 'uz', 'vx', 'vy', 'vz', 'xfem'])
        self.assertEqual(state_variable_names, SVAR_NAMES)

    #==============================================================================
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
        # almost equal, just down to float vs decimal repr
        self.assertAlmostEqual(answer['stress[sy]']['data'][0,0,0], -5373.5317, delta = 1e-4)


class ParallelTests:
    class ParallelSingleStateFile(unittest.TestCase):

        #==============================================================================
        def test_invalid_inputs(self):
            """Testing invalid inputs"""
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 4, labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(4, 'node', labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', material=9, labels = 4, states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 'cat', states = 300)
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 'cat')
            with self.assertRaises(MiliPythonError):
                self.mili.query(['nodpos[ux]'], 'node', labels = 4, states = 3, ips = 'cat')
            with self.assertRaises(TypeError):
                # class_name should be class_sname
                self.mili.query(svar_names="sx", class_name="brick")
            with self.assertRaises(TypeError):
                self.mili.query("sx")

        #==============================================================================
        def test_metadata(self):
            metadata = self.mili.metadata()
            EXPECTED = Metadata(
                code_name = "",
                username = "jdurren",
                job_id = "",
                nprocs = 8,
                date = "Wed Mar  3 09:01:15 2021",
                host_name = "rzgenie11",
                library_version = "V20_02"
            )
            for m in metadata:
                np.testing.assert_equal(m, EXPECTED)

        #==============================================================================
        def test_nodes_getter(self):
            """Testing the getNodes() method of the Mili class."""
            nodes = self.mili.nodes()

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

        #==============================================================================
        def test_statemaps_getter(self):
            """Testing the getStateMaps() method of the Mili class"""
            state_maps = self.mili.state_maps()
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            PROCS = 8
            self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
            for state_map in state_maps:
                self.assertEqual(len(state_map), STATE_COUNT)
                self.assertEqual(state_map[0].time, FIRST_STATE)
                self.assertEqual(state_map[-1].time, LAST_STATE)

        #==============================================================================
        def test_times(self):
            """Testing the times() method of the Mili class"""
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            PROCS = 8
            ptimes = self.mili.times()
            self.assertEqual(len(ptimes), PROCS)  # One entry in list for each processor
            for times in ptimes:
                self.assertEqual(STATE_COUNT, len(times))
                self.assertEqual(FIRST_STATE, times[0])
                self.assertEqual(LAST_STATE, times[-1])

        #==============================================================================
        def test_element_sets(self):
            """Testing the element_sets() method of the MiliDatabase class."""
            elem_sets = self.mili.element_sets()
            self.assertEqual( len(elem_sets), 8 )

            self.assertEqual( elem_sets[0], {} )
            self.assertEqual( elem_sets[1], {} )
            self.assertEqual( elem_sets[2], {"es_1": [1,2,3,4,4], "es_3": [1,2,2]} )
            self.assertEqual( elem_sets[3], {"es_3": [1,2,2]} )
            self.assertEqual( elem_sets[4], {"es_1": [1,2,3,4,4]} )
            self.assertEqual( elem_sets[5], {"es_1": [1,2,3,4,4], "es_3": [1,2,2]} )
            self.assertEqual( elem_sets[6], {"es_1": [1,2,3,4,4]} )
            self.assertEqual( elem_sets[7], {"es_1": [1,2,3,4,4]} )

        #==============================================================================
        def test_integration_points(self):
            """Testing the integration_points() method of the MiliDatabase class."""
            int_points = self.mili.integration_points()
            self.assertEqual( len(int_points), 8 )

            self.assertEqual( int_points[0], {} )
            self.assertEqual( int_points[1], {} )
            self.assertEqual( int_points[2], {"1": [1,2,3,4], "3": [1,2]} )
            self.assertEqual( int_points[3], {"3": [1,2]} )
            self.assertEqual( int_points[4], {"1": [1,2,3,4]} )
            self.assertEqual( int_points[5], {"1": [1,2,3,4], "3": [1,2]} )
            self.assertEqual( int_points[6], {"1": [1,2,3,4]} )
            self.assertEqual( int_points[7], {"1": [1,2,3,4]} )

        #==============================================================================
        def test_labels_getter(self):
            result = self.mili.labels()
            goal = [
                {
                'brick': np.array([ 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 66,  67,  68,  70,  71,  72,  74,  75,  76,  78,  79,  80,  82, 83,  84,  86,  87,  88,  90,  91,  92,  94,  95,  96,  98,  99, 100, 102, 103, 104, 106, 107, 108, 111, 112], dtype=np.int32)
                },
                {
                'brick': np.array([ 12, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 90,  91,  94,  95,  98,  99, 100, 102, 103, 104, 106, 107, 108, 110, 111, 112, 114, 115, 116, 118, 119, 120, 122, 123, 124, 126, 127, 128, 130, 131, 132, 134, 135, 136, 138, 139, 140, 142, 143, 144], dtype=np.int32)
                },
                {
                'beam': np.array([ 5, 6 ], dtype=np.int32),
                'brick': np.array([ 21, 23, 24 ], dtype=np.int32),
                'cseg': np.array([ 8, 9, 12, 20, 21, 24 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([  4,   6,  13,  14,  15,  58,  59,  60,  63,  64, 101, 102, 105, 106, 109, 110, 117, 118, 121, 122, 125, 126, 137, 138, 141, 142], dtype=np.int32),
                'shell': np.array([ 9, 11, 12 ], dtype=np.int32)
                },
                {
                'brick': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
                'cseg': np.array([ 1,  2,  3,  4,  5,  6, 13, 14, 15, 16, 17, 18 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 12,  13,  14,  15,  49,  50,  51,  52,  53,  54,  55,  56,  65, 66,  69,  70,  73,  74,  77,  78,  81,  82,  85,  86,  89,  90, 93,  94,  97,  98, 101, 102, 105, 106, 109, 110], dtype=np.int32),
                'shell': np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
                },
                {
                'beam': np.array([ 2, 3, 12, 13, 17, 22, 23, 24, 25, 27, 28, 32, 33, 34 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 3,  4,  5,  6,  7, 12, 13, 16, 21, 22, 25, 29, 30, 31, 32, 33, 34, 37, 38, 39], dtype=np.int32)
                },
                {
                'beam': np.array([ 1, 4 ], dtype=np.int32),
                'brick': np.array([ 19, 20, 22 ], dtype=np.int32),
                'cseg': np.array([ 7, 10, 11, 19, 22, 23 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([  1,   2,   3,  12,  13,  57,  58,  59,  61,  62,  63,  97,  98, 101, 102, 113, 114, 117, 118, 121, 122, 129, 130, 133, 134, 137, 138], dtype=np.int32),
                'shell': np.array([ 7, 8, 10 ], dtype=np.int32)
                },
                {
                'beam': np.array([ 11, 18, 19, 20, 21, 29, 30, 31, 37, 38, 39, 40, 41, 42 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 8, 10, 14, 15, 16, 20, 25, 26, 27, 28, 34, 35, 36, 41, 42, 43, 44, 45], dtype=np.int32)
                },
                {
                'beam': np.array([ 7, 8, 9, 10, 14, 15, 16, 26, 35, 36, 43, 44, 45, 46 ], dtype=np.int32),
                'glob': np.array([1], dtype=np.int32),
                'mat': np.array([1, 2, 3, 4, 5], dtype=np.int32),
                'node': np.array([ 1,  9, 11, 16, 17, 18, 19, 20, 22, 23, 24, 32, 39, 40, 45, 46, 47, 48], dtype=np.int32)
                }
                ]

            for pr, gr in zip(result, goal):
                for sname, labels in pr.items():
                    self.assertTrue( sname in gr.keys( ) )
                    np.testing.assert_equal( labels, gr[sname] )

        #==============================================================================
        def test_reload_state_maps(self):
            """Test the reload_state_maps method of the Mili Class."""
            # Just make sure it doesn't cause Exception
            self.mili.reload_state_maps()
            state_maps = self.mili.state_maps()
            FIRST_STATE = 0.0
            LAST_STATE = 0.0010000000474974513
            STATE_COUNT = 101
            PROCS = 8
            self.assertEqual(len(state_maps), PROCS)  # One entry in list for each processor
            for state_map in state_maps:
                self.assertEqual(len(state_map), STATE_COUNT)
                self.assertEqual(state_map[0].time, FIRST_STATE)
                self.assertEqual(state_map[-1].time, LAST_STATE)

        #==============================================================================
        def test_faces(self):
            with self.assertRaises(MiliPythonError):
                self.mili.faces("brickkkkk", 1)
            with self.assertRaises(MiliPythonError):
                self.mili.faces("shell", 1)
            with self.assertRaises(MiliPythonError):
                self.mili.faces("brick", 100)

            faces = self.mili.faces("brick", 1)
            np.testing.assert_equal(faces[3][1], [81, 85, 86, 82])
            np.testing.assert_equal(faces[3][2], [85, 69, 70, 86])
            np.testing.assert_equal(faces[3][3], [65, 66, 70, 69])
            np.testing.assert_equal(faces[3][4], [81, 82, 66, 65])
            np.testing.assert_equal(faces[3][5], [66, 82, 86, 70])
            np.testing.assert_equal(faces[3][6], [65, 69, 85, 81])

            faces = self.mili.faces("brick", 20)
            np.testing.assert_equal(faces[5][1], [129, 133, 134, 130])
            np.testing.assert_equal(faces[5][2], [133, 117, 118, 134])
            np.testing.assert_equal(faces[5][3], [113, 114, 118, 117])
            np.testing.assert_equal(faces[5][4], [129, 130, 114, 113])
            np.testing.assert_equal(faces[5][5], [114, 130, 134, 118])
            np.testing.assert_equal(faces[5][6], [113, 117, 133, 129])

            faces = self.mili.faces("brick", 32)
            np.testing.assert_equal(faces[1][1], [131, 135, 136, 132])
            np.testing.assert_equal(faces[1][2], [135, 119, 120, 136])
            np.testing.assert_equal(faces[1][3], [115, 116, 120, 119])
            np.testing.assert_equal(faces[1][4], [131, 132, 116, 115])
            np.testing.assert_equal(faces[1][5], [116, 132, 136, 120])
            np.testing.assert_equal(faces[1][6], [115, 119, 135, 131])

        #==============================================================================
        def test_material_numbers(self):
            mat_nums = self.mili.material_numbers()
            np.testing.assert_equal(mat_nums[0], [2])
            np.testing.assert_equal(mat_nums[1], [2])
            np.testing.assert_equal(mat_nums[2], [1,2,3,4,5])
            np.testing.assert_equal(mat_nums[3], [2,3,4,5])
            np.testing.assert_equal(mat_nums[4], [1])
            np.testing.assert_equal(mat_nums[5], [1,2,3,4,5])
            np.testing.assert_equal(mat_nums[6], [1])
            np.testing.assert_equal(mat_nums[7], [1])

        #==============================================================================
        def test_material_classes(self):
            mat_classes = self.mili.material_classes(1)
            self.assertEqual(mat_classes[0], [])
            self.assertEqual(mat_classes[1], [])
            self.assertEqual(mat_classes[2], ["beam"])
            self.assertEqual(mat_classes[3], [])
            self.assertEqual(mat_classes[4], ["beam"])
            self.assertEqual(mat_classes[5], ["beam"])
            self.assertEqual(mat_classes[6], ["beam"])
            self.assertEqual(mat_classes[7], ["beam"])

            mat_classes = self.mili.material_classes(2)
            self.assertEqual(mat_classes[0], ["brick"])
            self.assertEqual(mat_classes[1], ["brick"])
            self.assertEqual(mat_classes[2], ["brick"])
            self.assertEqual(mat_classes[3], ["brick"])
            self.assertEqual(mat_classes[4], [])
            self.assertEqual(mat_classes[5], ["brick"])
            self.assertEqual(mat_classes[6], [])
            self.assertEqual(mat_classes[7], [])

            mat_classes = self.mili.material_classes(3)
            self.assertEqual(mat_classes[0], [])
            self.assertEqual(mat_classes[1], [])
            self.assertEqual(mat_classes[2], ["shell"])
            self.assertEqual(mat_classes[3], ["shell"])
            self.assertEqual(mat_classes[4], [])
            self.assertEqual(mat_classes[5], ["shell"])
            self.assertEqual(mat_classes[6], [])
            self.assertEqual(mat_classes[7], [])

            mat_classes = self.mili.material_classes(4)
            self.assertEqual(mat_classes[0], [])
            self.assertEqual(mat_classes[1], [])
            self.assertEqual(mat_classes[2], ["cseg"])
            self.assertEqual(mat_classes[3], ["cseg"])
            self.assertEqual(mat_classes[4], [])
            self.assertEqual(mat_classes[5], ["cseg"])
            self.assertEqual(mat_classes[6], [])
            self.assertEqual(mat_classes[7], [])

            mat_classes = self.mili.material_classes(5)
            self.assertEqual(mat_classes[0], [])
            self.assertEqual(mat_classes[1], [])
            self.assertEqual(mat_classes[2], ["cseg"])
            self.assertEqual(mat_classes[3], ["cseg"])
            self.assertEqual(mat_classes[4], [])
            self.assertEqual(mat_classes[5], ["cseg"])
            self.assertEqual(mat_classes[6], [])
            self.assertEqual(mat_classes[7], [])

            with self.assertRaises(MiliPythonError):
                mat_classes = self.mili.material_classes(None)

        #==============================================================================
        def test_classes_of_state_variable(self):
            sx_classes = self.mili.classes_of_state_variable('sx')
            self.assertEqual(sx_classes[0], ["brick"])
            self.assertEqual(sx_classes[1], ["brick"])
            self.assertEqual(set(sx_classes[2]), set(["beam", "shell", "brick"]))
            self.assertEqual(set(sx_classes[3]), set(["shell", "brick"]))
            self.assertEqual(sx_classes[4], ["beam"])
            self.assertEqual(set(sx_classes[5]), set(["beam", "shell", "brick"]))
            self.assertEqual(sx_classes[6], ["beam"])
            self.assertEqual(sx_classes[7], ["beam"])

            uy_classes = self.mili.classes_of_state_variable('uy')
            self.assertEqual(uy_classes[0], ["node"])
            self.assertEqual(uy_classes[1], ["node"])
            self.assertEqual(uy_classes[2], ["node"])
            self.assertEqual(uy_classes[3], ["node"])
            self.assertEqual(uy_classes[4], ["node"])
            self.assertEqual(uy_classes[5], ["node"])
            self.assertEqual(uy_classes[6], ["node"])
            self.assertEqual(uy_classes[7], ["node"])

            axf_classes = self.mili.classes_of_state_variable('axf')
            self.assertEqual(axf_classes[0], [])
            self.assertEqual(axf_classes[1], [])
            self.assertEqual(axf_classes[2], ["beam"])
            self.assertEqual(axf_classes[3], [])
            self.assertEqual(axf_classes[4], ["beam"])
            self.assertEqual(axf_classes[5], ["beam"])
            self.assertEqual(axf_classes[6], ["beam"])
            self.assertEqual(axf_classes[7], ["beam"])

        #==============================================================================
        def test_materials_of_class_name(self):
            """Test the materials_of_class_name method of Mili Class."""
            brick_mats = self.mili.materials_of_class_name("brick")
            beam_mats = self.mili.materials_of_class_name("beam")
            shell_mats = self.mili.materials_of_class_name("shell")
            cseg_mats = self.mili.materials_of_class_name("cseg")

            self.assertEqual( brick_mats[0].size, 11 )
            self.assertEqual( brick_mats[1].size, 13 )
            self.assertEqual( brick_mats[2].size, 3 )
            self.assertEqual( brick_mats[3].size, 6 )
            self.assertEqual( brick_mats[4].size, 0 )
            self.assertEqual( brick_mats[5].size, 3 )
            self.assertEqual( brick_mats[6].size, 0 )
            self.assertEqual( brick_mats[7].size, 0 )
            np.testing.assert_equal( np.unique(brick_mats[0]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_mats[1]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_mats[2]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_mats[3]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_mats[5]), np.array([2]) )

            self.assertEqual( beam_mats[0].size, 0 )
            self.assertEqual( beam_mats[1].size, 0 )
            self.assertEqual( beam_mats[2].size, 2 )
            self.assertEqual( beam_mats[3].size, 0 )
            self.assertEqual( beam_mats[4].size, 14 )
            self.assertEqual( beam_mats[5].size, 2 )
            self.assertEqual( beam_mats[6].size, 14 )
            self.assertEqual( beam_mats[7].size, 14 )
            np.testing.assert_equal( np.unique(beam_mats[2]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_mats[4]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_mats[5]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_mats[6]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_mats[7]), np.array([1]) )

            self.assertEqual( shell_mats[0].size, 0 )
            self.assertEqual( shell_mats[1].size, 0 )
            self.assertEqual( shell_mats[2].size, 3 )
            self.assertEqual( shell_mats[3].size, 6 )
            self.assertEqual( shell_mats[4].size, 0 )
            self.assertEqual( shell_mats[5].size, 3 )
            self.assertEqual( shell_mats[6].size, 0 )
            self.assertEqual( shell_mats[7].size, 0 )
            np.testing.assert_equal( np.unique(shell_mats[2]), np.array([3]) )
            np.testing.assert_equal( np.unique(shell_mats[3]), np.array([3]) )
            np.testing.assert_equal( np.unique(shell_mats[5]), np.array([3]) )

            self.assertEqual( cseg_mats[0].size, 0 )
            self.assertEqual( cseg_mats[1].size, 0 )
            self.assertEqual( cseg_mats[2].size, 6 )
            self.assertEqual( cseg_mats[3].size, 12 )
            self.assertEqual( cseg_mats[4].size, 0 )
            self.assertEqual( cseg_mats[5].size, 6 )
            self.assertEqual( cseg_mats[6].size, 0 )
            self.assertEqual( cseg_mats[7].size, 0 )
            np.testing.assert_equal( cseg_mats[2], np.array([4, 4, 4, 5, 5, 5]) )
            np.testing.assert_equal( cseg_mats[3], np.array([4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5]) )
            np.testing.assert_equal( cseg_mats[5], np.array([4, 4, 4, 5, 5, 5]))

        #==============================================================================
        def test_parts_of_class_name(self):
            """Test the parts_of_class_name method of Mili Class."""
            brick_parts = self.mili.parts_of_class_name("brick")
            beam_parts = self.mili.parts_of_class_name("beam")
            shell_parts = self.mili.parts_of_class_name("shell")
            cseg_parts = self.mili.parts_of_class_name("cseg")

            self.assertEqual( brick_parts[0].size, 11 )
            self.assertEqual( brick_parts[1].size, 13 )
            self.assertEqual( brick_parts[2].size, 3 )
            self.assertEqual( brick_parts[3].size, 6 )
            self.assertEqual( brick_parts[4].size, 0 )
            self.assertEqual( brick_parts[5].size, 3 )
            self.assertEqual( brick_parts[6].size, 0 )
            self.assertEqual( brick_parts[7].size, 0 )
            np.testing.assert_equal( np.unique(brick_parts[0]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_parts[1]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_parts[2]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_parts[3]), np.array([2]) )
            np.testing.assert_equal( np.unique(brick_parts[5]), np.array([2]) )

            self.assertEqual( beam_parts[0].size, 0 )
            self.assertEqual( beam_parts[1].size, 0 )
            self.assertEqual( beam_parts[2].size, 2 )
            self.assertEqual( beam_parts[3].size, 0 )
            self.assertEqual( beam_parts[4].size, 14 )
            self.assertEqual( beam_parts[5].size, 2 )
            self.assertEqual( beam_parts[6].size, 14 )
            self.assertEqual( beam_parts[7].size, 14 )
            np.testing.assert_equal( np.unique(beam_parts[2]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_parts[4]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_parts[5]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_parts[6]), np.array([1]) )
            np.testing.assert_equal( np.unique(beam_parts[7]), np.array([1]) )

            self.assertEqual( shell_parts[0].size, 0 )
            self.assertEqual( shell_parts[1].size, 0 )
            self.assertEqual( shell_parts[2].size, 3 )
            self.assertEqual( shell_parts[3].size, 6 )
            self.assertEqual( shell_parts[4].size, 0 )
            self.assertEqual( shell_parts[5].size, 3 )
            self.assertEqual( shell_parts[6].size, 0 )
            self.assertEqual( shell_parts[7].size, 0 )
            np.testing.assert_equal( np.unique(shell_parts[2]), np.array([3]) )
            np.testing.assert_equal( np.unique(shell_parts[3]), np.array([3]) )
            np.testing.assert_equal( np.unique(shell_parts[5]), np.array([3]) )

            self.assertEqual( cseg_parts[0].size, 0 )
            self.assertEqual( cseg_parts[1].size, 0 )
            self.assertEqual( cseg_parts[2].size, 6 )
            self.assertEqual( cseg_parts[3].size, 12 )
            self.assertEqual( cseg_parts[4].size, 0 )
            self.assertEqual( cseg_parts[5].size, 6 )
            self.assertEqual( cseg_parts[6].size, 0 )
            self.assertEqual( cseg_parts[7].size, 0 )
            np.testing.assert_equal( np.unique(cseg_parts[2]), np.array([1]) )
            np.testing.assert_equal( np.unique(cseg_parts[3]), np.array([1]) )
            np.testing.assert_equal( np.unique(cseg_parts[5]), np.array([1]) )

        #==============================================================================
        def test_mesh_object_classes_getter(self):
            """Test the mesh_object_classes() method of Mili Class."""
            MO_classes = self.mili._mili.mesh_object_classes()

            # Test 0th processor
            mo_classes = MO_classes[0]

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
            self.assertEqual(node_class.elem_qty, 35)
            self.assertEqual(node_class.idents_exist, True)

            # brick class
            brick_class = mo_classes["brick"]
            self.assertEqual(brick_class.mesh_id, 0)
            self.assertEqual(brick_class.short_name, "brick")
            self.assertEqual(brick_class.long_name, "Bricks")
            self.assertEqual(brick_class.sclass, Superclass.M_HEX)
            self.assertEqual(brick_class.elem_qty, 11)
            self.assertEqual(brick_class.idents_exist, True)

            # Test processor 5
            mo_classes = MO_classes[5]

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
            self.assertEqual(node_class.elem_qty, 27)
            self.assertEqual(node_class.idents_exist, True)

            # brick class
            brick_class = mo_classes["brick"]
            self.assertEqual(brick_class.mesh_id, 0)
            self.assertEqual(brick_class.short_name, "brick")
            self.assertEqual(brick_class.long_name, "Bricks")
            self.assertEqual(brick_class.sclass, Superclass.M_HEX)
            self.assertEqual(brick_class.elem_qty, 3)
            self.assertEqual(brick_class.idents_exist, True)

            # shell class
            shell_class = mo_classes["shell"]
            self.assertEqual(shell_class.mesh_id, 0)
            self.assertEqual(shell_class.short_name, "shell")
            self.assertEqual(shell_class.long_name, "Shells")
            self.assertEqual(shell_class.sclass, Superclass.M_QUAD)
            self.assertEqual(shell_class.elem_qty, 3)
            self.assertEqual(shell_class.idents_exist, True)

            # cseg class
            cseg_class = mo_classes["cseg"]
            self.assertEqual(cseg_class.mesh_id, 0)
            self.assertEqual(cseg_class.short_name, "cseg")
            self.assertEqual(cseg_class.long_name, "Contact Segment")
            self.assertEqual(cseg_class.sclass, Superclass.M_QUAD)
            self.assertEqual(cseg_class.elem_qty, 6)

        #==============================================================================
        def test_statevariables_getter(self):
            """Testing the getStateVariables() method of the Mili class."""
            result = self.mili._mili.state_variables()
            goal = [159,126,134,134,126,134,126,126]
            result = [len(rr) for rr in result]
            self.assertEqual( result, goal )

        #==============================================================================
        def test_all_labels_of_material(self):
            """Testing whether the element numbers associated with a material are correct"""
            answer = self.mili.all_labels_of_material('es_13')
            np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
            np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
            np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

            answer = self.mili.all_labels_of_material('3')
            np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
            np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
            np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

            answer = self.mili.all_labels_of_material(3)
            np.testing.assert_equal(answer[2]['shell'],np.array([9,11,12], dtype=np.int32))
            np.testing.assert_equal(answer[3]['shell'],np.array([1,2,3,4,5,6], dtype=np.int32))
            np.testing.assert_equal(answer[5]['shell'],np.array([7,8,10], dtype=np.int32))

        #==============================================================================
        def test_nodes_material(self):
            """Testing what nodes are associated with a material"""
            answer = self.mili.nodes_of_material('es_1')
            node_labels = np.unique( np.concatenate((*answer,)) )
            np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

            answer = self.mili.nodes_of_material('1')
            node_labels = np.unique( np.concatenate((*answer,)) )
            np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

            answer = self.mili.nodes_of_material(1)
            node_labels = np.unique( np.concatenate((*answer,)) )
            np.testing.assert_equal( node_labels, np.arange(1, 49, dtype = np.int32) )

        #==============================================================================
        def test_nodes_label(self):
            """Testing what nodes are associated with a label"""
            answer = self.mili.nodes_of_elems('brick', 1)
            np.testing.assert_equal( answer[3][0][0,:], np.array([65,81,85,69,66,82,86,70],dtype=np.int32) )

        #==============================================================================
        def test_state_variable(self):
            """Testing accessing a variable at a given state"""
            answer = self.mili.query( 'matcgx', 'mat', labels = [1,2], states = 3 )
            self.assertEqual(answer[0]['matcgx']['title'], "C.G. X-Position")
            self.assertEqual(answer[0]['matcgx']['class_name'], "mat")
            np.testing.assert_equal( answer[0]['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32 ) )

            answer = self.mili.query( 'matcgx', 'mat', labels = [1,2], states = -99 )
            np.testing.assert_equal( answer[0]['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32 ) )

        #==============================================================================
        def test_node_attributes(self):
            """Testing accessing accessing node attributes -> this is a vector component
            Tests both ways of accessing vector components (using brackets vs not)
            *Note* this is another case of state variable
            """
            answer = self.mili.query( 'nodpos[ux]', 'node', labels = 70, states = 3 )
            self.assertEqual(answer[3]['nodpos[ux]']['title'], "Node Position")
            self.assertEqual(answer[3]['nodpos[ux]']['class_name'], "node")
            self.assertEqual(answer[3]['nodpos[ux]']['layout']['components'], ['ux'])
            self.assertEqual(answer[3]['nodpos[ux]']['layout']['labels'], [70])
            self.assertEqual(answer[3]['nodpos[ux]']['layout']['states'], [3])
            np.testing.assert_allclose(answer[3]['nodpos[ux]']['layout']['times'], [2.0e-05])
            self.assertEqual( answer[3]['nodpos[ux]']['data'][0,0,0], 0.4330127537250519)

            answer = self.mili.query( 'ux', 'node', labels = 70, states = 3 )
            self.assertEqual(answer[3]['ux']['title'], "X Position")
            self.assertEqual(answer[3]['ux']['class_name'], "node")
            self.assertEqual(answer[3]['ux']['layout']['components'], ['ux'])
            self.assertEqual(answer[3]['ux']['layout']['labels'], [70])
            self.assertEqual(answer[3]['ux']['layout']['states'], [3])
            np.testing.assert_allclose(answer[3]['ux']['layout']['times'], [2.0e-05])
            self.assertEqual( answer[3]['ux']['data'][0,0,0], 0.4330127537250519)

            answer = self.mili.query( 'nodpos[ux]', 'node', labels = 70, states = -99 )
            self.assertEqual( answer[3]['nodpos[ux]']['data'][0,0,0], 0.4330127537250519)
            answer = self.mili.query( 'ux', 'node', labels = 70, states = -99 )
            self.assertEqual( answer[3]['ux']['data'][0,0,0], 0.4330127537250519)

        #==============================================================================
        def test_state_variable_vector(self):
            """Testing the accessing of a vector, in this case node position"""
            answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
            self.assertEqual(answer[3]['nodpos']['title'], 'Node Position')
            self.assertEqual(answer[3]['nodpos']['class_name'], 'node')
            self.assertEqual(answer[3]['nodpos']['layout']['components'], ['ux', 'uy', 'uz'])
            np.testing.assert_equal(answer[3]['nodpos']['layout']['labels'], [70])
            np.testing.assert_equal(answer[3]['nodpos']['layout']['states'], [4])
            np.testing.assert_allclose(answer[3]['nodpos']['layout']['times'], [3.0e-05])
            np.testing.assert_equal(answer[3]['nodpos']['data'][0,0,:], np.array( [0.4330127537250519, 0.2500000596046448, 2.436666965484619], dtype = np.float32 ) )

        #==============================================================================
        def test_query_material(self):
            """Test querying by material name and number"""
            answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
            num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
            self.assertEqual( num_labels, 36 )

            answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
            num_labels = sum( pansw['sx']['layout'].get( 'labels', np.empty([0],dtype=np.int32) ).size for pansw in answer )
            self.assertEqual( num_labels, 36 )

        #==============================================================================
        def test_state_variable_vector_array(self):
            """Test accessing a vector array"""
            answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
            self.assertEqual( answer[2]['stress']['title'], 'Stress')
            self.assertEqual( answer[2]['stress']['class_name'], 'beam')
            self.assertEqual( answer[2]['stress']['layout']['components'], ['sx ipt. 2', 'sy ipt. 2', 'sz ipt. 2', 'sxy ipt. 2', 'syz ipt. 2', 'szx ipt. 2'] )
            np.testing.assert_equal(answer[2]['stress']['layout']['labels'], [5])
            np.testing.assert_equal(answer[2]['stress']['layout']['states'], [21,22])
            np.testing.assert_allclose(answer[2]['stress']['layout']['times'], [2.0e-04, 2.1e-04])
            np.testing.assert_equal( answer[2]['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307] ], dtype = np.float32 ) )

            answer = self.mili.query('stress', 'beam', labels = 5, states = [-81,-80], ips = 2 )
            self.assertEqual( answer[2]['stress']['title'], 'Stress')
            self.assertEqual( answer[2]['stress']['class_name'], 'beam')
            self.assertEqual( answer[2]['stress']['layout']['components'], ['sx ipt. 2', 'sy ipt. 2', 'sz ipt. 2', 'sxy ipt. 2', 'syz ipt. 2', 'szx ipt. 2'] )
            np.testing.assert_equal(answer[2]['stress']['layout']['labels'], [5])
            np.testing.assert_equal(answer[2]['stress']['layout']['states'], [21,22])
            np.testing.assert_allclose(answer[2]['stress']['layout']['times'], [2.0e-04, 2.1e-04])
            np.testing.assert_equal( answer[2]['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307] ], dtype = np.float32 ) )

        #==============================================================================
        def test_state_variable_vector_array_component(self):
            """Test accessing a vector array component"""
            answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 70, ips = 2 )
            self.assertEqual( answer[2]['stress[sy]']['title'], 'Stress')
            self.assertEqual( answer[2]['stress[sy]']['class_name'], 'beam')
            self.assertEqual( answer[2]['stress[sy]']['layout']['components'], ['sy ipt. 2'] )
            np.testing.assert_equal(answer[2]['stress[sy]']['layout']['labels'], [5])
            np.testing.assert_equal(answer[2]['stress[sy]']['layout']['states'], [70])
            np.testing.assert_allclose(answer[2]['stress[sy]']['layout']['times'], [6.9e-04])
            self.assertEqual(answer[2]['stress[sy]']['data'][0,0,0], -5373.53173828125)

        #==============================================================================
        def test_query_glob_results(self):
            """Test querying for results for M_MESH ("glob") element class."""
            answer = self.mili.query("he", "glob", states=[22])
            self.assertEqual( answer[0]['he']['title'], 'Hourglass Energy')
            self.assertEqual( answer[0]['he']['class_name'], 'glob')
            self.assertEqual( answer[0]['he']['layout']['components'], ['he'] )
            np.testing.assert_equal( answer[0]['he']['layout']['labels'], [1] )
            np.testing.assert_equal( answer[0]['he']['layout']['states'], [22] )
            np.testing.assert_allclose( answer[0]['he']['layout']['times'], [2.1e-04] )
            self.assertAlmostEqual( answer[0]["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

            answer = self.mili.query("bve", "glob", states=[22])
            self.assertEqual( answer[0]['bve']['title'], 'Bulk Vis. Energy')
            self.assertEqual( answer[0]['bve']['class_name'], 'glob')
            self.assertEqual( answer[0]['bve']['layout']['components'], ['bve'] )
            np.testing.assert_equal( answer[0]['bve']['layout']['labels'], [1] )
            np.testing.assert_equal( answer[0]['bve']['layout']['states'], [22] )
            np.testing.assert_allclose( answer[0]['bve']['layout']['times'], [2.1e-04] )
            self.assertAlmostEqual( answer[0]["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

            answer = self.mili.query("te", "glob", states=[22])
            self.assertEqual( answer[0]['te']['title'], 'Total Energy')
            self.assertEqual( answer[0]['te']['class_name'], 'glob')
            self.assertEqual( answer[0]['te']['layout']['components'], ['te'] )
            np.testing.assert_equal( answer[0]['te']['layout']['labels'], [1] )
            np.testing.assert_equal( answer[0]['te']['layout']['states'], [22] )
            np.testing.assert_allclose( answer[0]['te']['layout']['times'], [2.1e-04] )
            self.assertAlmostEqual( answer[0]["te"]["data"][0,0,0], 1629.718, delta=1e-4)

        def test_cummin(self):
            """Test cumulative min derived variable."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", labels=[1,7,11], ips=[1], modifier=ResultModifier.CUMMIN)
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "beam")
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "cummin")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,11,7])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

            # Beam 1 has max of 0.0 across all states
            np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

            # Beam 7
            np.testing.assert_allclose(result[svar_name]['data'][0:21,2,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][21,2,:], -2.864564, rtol=2.0e-07)
            np.testing.assert_allclose(result[svar_name]['data'][22,2,:], -4.125320)
            np.testing.assert_allclose(result[svar_name]['data'][48:60,2,:], -3.7649155e+02)
            np.testing.assert_allclose(result[svar_name]['data'][63:65,2,:], -6.132141e+02)
            np.testing.assert_allclose(result[svar_name]['data'][65:82,2,:], -6.723166e+02)
            np.testing.assert_allclose(result[svar_name]['data'][82:,2,:], -7.151691e+02)

            # Beam 11
            np.testing.assert_allclose(result[svar_name]['data'][:,1,:], 0.0)

            svar_name = "disp_x"
            result = self.mili.query(svar_name, "node", labels=[1,44], modifier=ResultModifier.CUMMIN)
            self.assertEqual(result[svar_name]["title"], "X Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "cummin")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_x"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,44])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

            # Node 1
            np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)
            # Node 44
            np.testing.assert_allclose(result[svar_name]['data'][:,1,:], 0.0)

        def test_cummax(self):
            """Test cumulative max derived variable."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", labels=[1,7,11], ips=[1], modifier=ResultModifier.CUMMAX)
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "beam")
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "cummax")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,11,7])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

            # Beam 1 has max of 0.0 across all states
            np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

            # Beam 7
            np.testing.assert_allclose(result[svar_name]['data'][0:35,2,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][35:,2,:], 6.722762)

            # Beam 11
            np.testing.assert_allclose(result[svar_name]['data'][0:21,1,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][21,1,:], 0.12996, rtol=4.0e-06)
            np.testing.assert_allclose(result[svar_name]['data'][22,1,:], 5.531850)
            np.testing.assert_allclose(result[svar_name]['data'][52,1,:], 2078.374)
            np.testing.assert_allclose(result[svar_name]['data'][53:,1,:], 2199.8823)

            svar_name = "disp_x"
            result = self.mili.query(svar_name, "node", labels=[1,44], modifier=ResultModifier.CUMMAX)
            self.assertEqual(result[svar_name]["title"], "X Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "cummax")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_x"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,44])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], np.arange(1,102))
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], np.arange(0.0,1.01e-03, 1.0e-05))

            # Node 1 has max of 0.0 across all states
            np.testing.assert_allclose(result[svar_name]['data'][:,0,:], 0.0)

            # Node 44
            np.testing.assert_allclose(result[svar_name]['data'][0:21,1,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][21,1,:], 4.011393e-05, rtol=2.0e-07)
            np.testing.assert_allclose(result[svar_name]['data'][22,1,:], 2.411008e-04)
            np.testing.assert_allclose(result[svar_name]['data'][57,1,:], 6.289059e-02)
            np.testing.assert_allclose(result[svar_name]['data'][58:,1,:], 6.300801e-02, rtol=10.0e-07)

        def test_query_min(self):
            """Test min query modifier."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.MIN)
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "min")
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "beam")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [5,6,6])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -1370.853882)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -840.760498)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.MIN)
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "min")
            self.assertEqual(result[svar_name]["title"], "Y Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,1,1])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.0)

        def test_query_min_dataframe(self):
            """Test min query modifier with as_dataframe=True."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.MIN)
            df = result[svar_name]
            np.testing.assert_equal(list(df.columns), ["min", "label"])
            np.testing.assert_equal(list(df.index), [21,22,23])

            np.testing.assert_equal(df["label"][21], 5)
            np.testing.assert_equal(df["label"][22], 6)
            np.testing.assert_equal(df["label"][23], 6)
            np.testing.assert_allclose(df["min"][21], 0.0)
            np.testing.assert_allclose(df["min"][22], -1370.853882)
            np.testing.assert_allclose(df["min"][23], -840.760498)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.MIN)
            df = result[svar_name]
            self.assertEqual(list(df.columns), ["min", "label"])
            self.assertEqual(list(df.index), [97,98,99])

            np.testing.assert_equal(df["label"][97], 1)
            np.testing.assert_equal(df["label"][98], 1)
            np.testing.assert_equal(df["label"][99], 1)
            np.testing.assert_allclose(df["min"][97], 0.0)
            np.testing.assert_allclose(df["min"][98], 0.0)
            np.testing.assert_allclose(df["min"][99], 0.0)

        def test_query_max(self):
            """Test max query modifier."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.MAX)
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "max")
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "beam")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [5,5,4])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 307.012909)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 41.389294)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.MAX)
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "max")
            self.assertEqual(result[svar_name]["title"], "Y Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [11,11,11])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.146183, rtol=4.0e-06)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.145862, rtol=10.0e-07)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.145216, rtol=10.0e-07)

        def test_query_max_dataframe(self):
            """Test max query modifier with as_dataframe=True."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.MAX)
            df = result[svar_name]
            np.testing.assert_equal(list(df.columns), ["max", "label"])
            np.testing.assert_equal(list(df.index), [21,22,23])

            np.testing.assert_equal(df["label"][21], 5)
            np.testing.assert_equal(df["label"][22], 5)
            np.testing.assert_equal(df["label"][23], 4)
            np.testing.assert_allclose(df["max"][21], 0.0)
            np.testing.assert_allclose(df["max"][22], 307.012909)
            np.testing.assert_allclose(df["max"][23], 41.389294)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.MAX)
            df = result[svar_name]
            self.assertEqual(list(df.columns), ["max", "label"])
            self.assertEqual(list(df.index), [97,98,99])

            np.testing.assert_equal(df["label"][97], 11)
            np.testing.assert_equal(df["label"][98], 11)
            np.testing.assert_equal(df["label"][99], 11)
            np.testing.assert_allclose(df["max"][97], 0.146183, rtol=4.0e-06)
            np.testing.assert_allclose(df["max"][98], 0.145862, rtol=10.0e-07)
            np.testing.assert_allclose(df["max"][99], 0.145216, rtol=10.0e-07)

        def test_query_average(self):
            """Test average query modifier."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], modifier=ResultModifier.AVERAGE)
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "average")
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "beam")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx ipt. 1"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.0)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -25.27041)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -23.06732)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], modifier=ResultModifier.AVERAGE)
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "average")
            self.assertEqual(result[svar_name]["title"], "Y Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.010688, rtol=3.0e-05)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.010649, rtol=5.0e-05)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.010595, rtol=5.0e-05)

        def test_query_average_dataframe(self):
            """Test average query modifier with as_dataframe=True."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "beam", states=[21,22,23], ips=[1], as_dataframe=True, modifier=ResultModifier.AVERAGE)
            df = result[svar_name]
            np.testing.assert_equal(list(df.columns), ["average"])
            np.testing.assert_equal(list(df.index), [21,22,23])
            np.testing.assert_allclose(df["average"][21], 0.0)
            np.testing.assert_allclose(df["average"][22], -25.27041)
            np.testing.assert_allclose(df["average"][23], -23.06732)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], as_dataframe=True, modifier=ResultModifier.AVERAGE)
            df = result[svar_name]
            self.assertEqual(list(df.columns), ["average"])
            self.assertEqual(list(df.index), [97,98,99])
            np.testing.assert_allclose(df["average"][97], 0.010688, rtol=3.0e-05)
            np.testing.assert_allclose(df["average"][98], 0.010649, rtol=5.0e-05)
            np.testing.assert_allclose(df["average"][99], 0.010595, rtol=5.0e-05)

        def test_query_median(self):
            """Test median query modifier."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], ips=[1], modifier=ResultModifier.MEDIAN)
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "median")
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "brick")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,2,3,4,5])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], -1.924881e-09, rtol=1.2e-07)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], -7.198997e+03)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], -5.880557e+03)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], modifier=ResultModifier.MEDIAN)
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "median")
            self.assertEqual(result[svar_name]["title"], "Y Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.038422, rtol=6.0e-06)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.038286, rtol=7.1e-06)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.038071, rtol=4.2e-06)

        def test_query_median_dataframe(self):
            """Test median query modifier with as_dataframe=True."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], as_dataframe=True, modifier=ResultModifier.MEDIAN)
            df = result[svar_name]
            np.testing.assert_equal(list(df.columns), ["median"])
            np.testing.assert_equal(list(df.index), [21,22,23])
            np.testing.assert_allclose(df["median"][21], -1.924881e-09, rtol=1.2e-07)
            np.testing.assert_allclose(df["median"][22], -7.198997e+03)
            np.testing.assert_allclose(df["median"][23], -5.880557e+03)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], as_dataframe=True, modifier=ResultModifier.MEDIAN)
            df = result[svar_name]
            self.assertEqual(list(df.columns), ["median"])
            self.assertEqual(list(df.index), [97,98,99])
            np.testing.assert_allclose(df["median"][97], 0.038422, rtol=6.0e-06)
            np.testing.assert_allclose(df["median"][98], 0.038286, rtol=7.1e-06)
            np.testing.assert_allclose(df["median"][99], 0.038071, rtol=4.1e-06)

        def test_query_stddev(self):
            """Test stddev query modifier."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], ips=[1], modifier=ResultModifier.STDDEV)
            self.assertEqual(result[svar_name]["source"], "primal")
            self.assertEqual(result[svar_name]["modifier"], "stddev")
            self.assertEqual(result[svar_name]["title"], "X Stress")
            self.assertEqual(result[svar_name]["class_name"], "brick")
            self.assertEqual(result[svar_name]["layout"]["components"], ["sx"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [21,22,23])
            np.testing.assert_equal(result[svar_name]["layout"]["labels"], [1,2,3,4,5])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [2.0e-04, 2.1e-04, 2.2e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 7.788077e-09, rtol=1.2e-07)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 9.055870e+03)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 2.214029e+03, rtol=1.2e-07)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], modifier=ResultModifier.STDDEV)
            self.assertEqual(result[svar_name]["source"], "derived")
            self.assertEqual(result[svar_name]["modifier"], "stddev")
            self.assertEqual(result[svar_name]["title"], "Y Displacement")
            self.assertEqual(result[svar_name]["class_name"], "node")
            self.assertEqual(result[svar_name]["layout"]["components"], ["disp_y"])
            np.testing.assert_equal(result[svar_name]["layout"]["states"], [97,98,99])
            np.testing.assert_allclose(result[svar_name]["layout"]["times"], [9.6e-04, 9.7e-04, 9.8e-04])

            np.testing.assert_allclose(result[svar_name]['data'][0,0,:], 0.025586, rtol=1.6e-05)
            np.testing.assert_allclose(result[svar_name]['data'][1,0,:], 0.025499, rtol=1.1e-05)
            np.testing.assert_allclose(result[svar_name]['data'][2,0,:], 0.025357, rtol=3.3e-06)

        def test_query_stddev_dataframe(self):
            """Test stddev query modifier with as_dataframe=True."""
            svar_name = "sx"
            result = self.mili.query(svar_name, "brick", states=[21,22,23], labels=[1,2,3,4,5], as_dataframe=True, modifier=ResultModifier.STDDEV)
            df = result[svar_name]
            np.testing.assert_equal(list(df.columns), ["stddev"])
            np.testing.assert_equal(list(df.index), [21,22,23])
            np.testing.assert_allclose(df["stddev"][21], 7.788077e-09, rtol=1.2e-07)
            np.testing.assert_allclose(df["stddev"][22], 9.055870e+03)
            np.testing.assert_allclose(df["stddev"][23], 2.214029e+03, rtol=1.2e-07)

            svar_name = "disp_y"
            result = self.mili.query(svar_name, "node", states=[97,98,99], labels=[1,23,40], as_dataframe=True, modifier=ResultModifier.STDDEV)
            df = result[svar_name]
            self.assertEqual(list(df.columns), ["stddev"])
            self.assertEqual(list(df.index), [97,98,99])
            np.testing.assert_allclose(df["stddev"][97], 0.025586, rtol=1.6e-05)
            np.testing.assert_allclose(df["stddev"][98], 0.025499, rtol=1.1e-05)
            np.testing.assert_allclose(df["stddev"][99], 0.025357, rtol=3.3e-06)

###############################################
# Tests for each parallel wrapper class
###############################################

class LoopWrapperParallelTests(ParallelTests.ParallelSingleStateFile):
    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( LoopWrapperParallelTests.file_name, suppress_parallel=True, merge_results=False )

class ServerWrapperParallelTests(ParallelTests.ParallelSingleStateFile):

    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = reader.open_database( ServerWrapperParallelTests.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()

class LoopWrapperContextManagerParallelTests(ParallelTests.ParallelSingleStateFile):
    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def run(self, result=None):
        with reader.open_database( LoopWrapperParallelTests.file_name, suppress_parallel=True, merge_results=False ) as mili:
            self.mili = mili
            super(LoopWrapperContextManagerParallelTests, self).run(result)

class ServerWrapperContextManagerParallelTests(ParallelTests.ParallelSingleStateFile):

    file_name = file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def run(self, result=None):
        with reader.open_database( ServerWrapperParallelTests.file_name, suppress_parallel=False, merge_results=False ) as mili:
            self.mili = mili
            super(ServerWrapperContextManagerParallelTests, self).run(result)

class SerialContextManagerTests(SharedSerialTests.SerialTests):
    file_name = os.path.join(dir_path,'data','serial','sstate','d3samp6.plt')

    def run(self, result=None):
        with reader.open_database( SerialSingleStateFile.file_name, suppress_parallel = True, merge_results=False ) as mili:
            self.mili = mili
            super(SerialContextManagerTests, self).run(result)

    # Tests unique to single state file
    #==============================================================================
    def test_state_variables(self):
        state_variable_names = set(self.mili._mili.state_variables().keys())
        SVAR_NAMES = set(['ke', 'ke_part', 'pe', 'he', 'bve', 'dre', 'stde', 'flde', 'tcon_damp_eng',
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
                          'dam', 'frac_strain', 'sand', 'cause'])
        self.assertEqual(state_variable_names, SVAR_NAMES)

    #==============================================================================
    def test_state_variable(self):
        answer = self.mili.query('matcgx', 'mat', labels = [1,2], states = 3 )
        self.assertEqual(answer['matcgx']['layout']['states'][0], 3)
        self.assertEqual(list(answer.keys()), ['matcgx'] )
        np.testing.assert_equal( answer['matcgx']['layout']['labels'], np.array( [ 1, 2 ], dtype = np.int32) )
        np.testing.assert_equal( answer['matcgx']['data'][0,:,:], np.array( [ [ 0.6021666526794434 ], [ 0.6706029176712036 ] ], dtype = np.float32) )

    #==============================================================================
    def test_node_attributes(self):
        answer = self.mili.query('nodpos[ux]', 'node', labels = 70, states = 3 )
        self.assertEqual(answer['nodpos[ux]']['layout']['labels'][0], 70)
        self.assertEqual(answer['nodpos[ux]']['data'][0], 0.4330127537250519 )

        answer = self.mili.query('ux', EntityType.NODE, labels = 70, states = 3 )
        self.assertEqual(answer['ux']['layout']['labels'][0], 70)
        self.assertEqual(answer['ux']['data'][0], 0.4330127537250519)

    #==============================================================================
    def test_query_material(self):
        answer = self.mili.query('sx', 'brick', material = 2, states = 37 )
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

        answer = self.mili.query('sx', 'brick', material = 'es_12', states = 37 )
        self.assertEqual(answer['sx']['layout']['labels'].size, 36)

    #==============================================================================
    def test_state_variable_vector(self):
        answer = self.mili.query('nodpos', 'node', labels = 70, states = 4 )
        np.testing.assert_equal( answer['nodpos']['data'][0,:,:], np.array( [ [ 0.4330127537250519, 0.2500000596046448, 2.436666965484619 ] ], dtype = np.float32 ) )

    #==============================================================================
    def test_state_variable_vector_array(self):
        answer = self.mili.query('stress', 'beam', labels = 5, states = [21,22], ips = 2 )
        np.testing.assert_equal( answer['stress']['data'][1,:,:], np.array([ [ -1018.4232177734375, -1012.2537231445312, -6.556616085617861e-07, 1015.3384399414062, 0.3263571858406067, -0.32636013627052307 ] ], dtype = np.float32 ) )

    #==============================================================================
    def test_state_variable_vector_array_component(self):
        answer = self.mili.query('stress[sy]', 'beam', labels = 5, states = 71, ips = 2 )
        self.assertEqual(answer['stress[sy]']['data'][0,0,0], -5545.70751953125)

    #==============================================================================
    def test_query_glob_results(self):
        """
        Test querying for results for M_MESH ("glob") element class.
        """
        answer = self.mili.query("he", "glob", states=[22])
        self.assertAlmostEqual( answer["he"]["data"][0,0,0], 3.0224223, delta=1e-7)

        answer = self.mili.query("bve", "glob", states=[22])
        self.assertAlmostEqual( answer["bve"]["data"][0,0,0], 2.05536485, delta=1e-7)

        answer = self.mili.query(GlobalStateVariables.TOTAL_ENERGY, EntityType.GLOBAL, states=[22])
        self.assertAlmostEqual( answer["te"]["data"][0,0,0], 1629.718, delta=1e-4)

if __name__ == "__main__":
    unittest.main()


