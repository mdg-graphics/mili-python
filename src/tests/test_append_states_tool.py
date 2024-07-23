#!/usr/bin/env python3

"""
SPDX-License-Identifier: (MIT)
"""
import os
import shutil
import unittest
import numpy as np
from mili import reader
from mili.append_states import AppendStatesTool

dir_path = os.path.dirname(os.path.realpath(__file__))

class TestAppendStateToolInvalidInput(unittest.TestCase):
    """Test the invalid input cases for the Append states tool."""

    def test_invalid_output_type(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["silo"]
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_output_mode(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "writ",
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_missing_output_basename(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "write",
        }
        with self.assertRaises(KeyError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_limit_states(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "limit_states_per_file": 0
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_limit_bytes(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "limit_bytes_per_file": 0
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_states(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": -1
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_missing_state_times_time_inc(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1
        }
        with self.assertRaises(KeyError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_time_inc(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "time_inc": 0.0
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_mismatched_states_state_times(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [1.0, 2.0]
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_append_state_times(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [0.0]
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_state_variable_missing_required_values(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [0.1],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3]
                        # Missing 'data'
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_class_name(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [0.1],

            "state_variables": {
                "does-not-exist": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0]
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_labels(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [0.1],

            "state_variables": {
                "shell": {
                    "sx": {
                        "labels": [20,21,22], # These labels don't exist
                        'data': [
                            [1.0, 2.0, 3.0]
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_int_point(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 1,
            "state_times": [0.1],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'int_point': 1, # This int_point doesn't exist
                        'data': [
                            [1.0, 2.0, 3.0]
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_data_1d_array(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "state_times": [0.1, 0.2],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3],
                        #'int_point': 1, # This int_point doesn't exist
                        'data': [1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0]
                        # 1D data array okay but should have length of 6
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_data_jagged_array(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "state_times": [0.1, 0.2],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0, 7.0]
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_data_states_dim(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "state_times": [0.1, 0.2],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            # There should be a second row of data here
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_data_results_dim(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "state_times": [0.1, 0.2],

            "state_variables": {
                "brick": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0], # These should each be 3 elements long
                            [1.0, 2.0],
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)

    def test_invalid_data_results_dim_int_points(self):
        append_states_spec = {
            "database_basename": os.path.join(dir_path,'data','serial','sstate','v3_d3samp6.plt'),
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "state_times": [0.1, 0.2],

            "state_variables": {
                "shell": {
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            # No int_point is provided so all should be here instead of just 1
                            [1.0, 2.0, 3.0],
                            [1.0, 2.0, 3.0],
                        ]
                    }
                }
            }
        }
        with self.assertRaises(ValueError):
            tool = AppendStatesTool(append_states_spec)


class TestAppendStatesToolSerial(unittest.TestCase):
    """Tests the AppendStatesTool object on an serial database."""
    data_path = os.path.join(dir_path,'data','serial','sstate')
    required_plot_files = ["v3_d3samp6.pltA", "v3_d3samp6.plt00"]
    base_name = 'v3_d3samp6.plt'
    new_base_name = 'copy_d3samp6.plt'

    def copy_required_plot_files(self, path, required_files, test_prefix):
        for file in required_files:
            fname = os.path.join(path, file)
            shutil.copyfile(fname, f"./{test_prefix}_{file}")

    def setUp(self):
        self.copy_required_plot_files(TestAppendStatesToolSerial.data_path,
                                      TestAppendStatesToolSerial.required_plot_files,
                                      self._testMethodName)

    def tearDown(self):
        # Only delete temp files if test passes.
        if self._outcome.success:
            for f in TestAppendStatesToolSerial.required_plot_files:
                os.remove(f"./{self._testMethodName}_{f}")
            if os.path.exists(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}A"):
                os.remove(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}A")
            if os.path.exists(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}00"):
                os.remove(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}00")
            if os.path.exists(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}01"):
                os.remove(f"./{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}01")

    def test_append_to_existing(self):
        db_name = f"{self._testMethodName}_{TestAppendStatesToolSerial.base_name}"
        append_states_spec = {
            "database_basename": db_name,
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "time_inc": 0.001,
            "limit_states_per_file": 2,

            "state_variables": {
                "node": {
                    "vy": {
                        'labels': [100, 101, 102],
                        'data': [
                            [100.0, 100.0, 100.0],
                            [200.0, 300.0, 400.0],
                        ]
                    }
                },
                "brick": {
                    # Scalar state variable (no int_points)
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable (no int_points)
                    "strain": {
                        'labels': [10, 11, 12],
                        'data': [
                            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                        ]
                    }
                },
                "shell": {
                    # Scalar state variable w/ SINGLE int_points
                    "edrate": {
                        "labels": [7, 8, 9],
                        'int_point': 1,
                        'data': [
                            [10.0, 20.0, 30.0],
                            [40.0, 50.0, 60.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_point)
                    "strain": {
                        'labels': [5,6],
                        'int_point': 2,
                        'data': [
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                        ]
                    }
                },
                "beam": {
                    # Scalar state variable (no int_points)
                    "axf": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_points)
                    "stress": {
                        'labels': [43],
                        # Data for 1 element with 4 integration points
                        'data': [
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                        ]
                    }
                },
            }
        }
        tool = AppendStatesTool(append_states_spec)
        tool.write_states()

        # Verify new database looks correct.
        db = reader.open_database( db_name )

        # Test state maps
        state_maps = db.state_maps()
        self.assertEqual( len(state_maps), 103 )
        self.assertEqual(state_maps[-3].file_number, 0)
        self.assertEqual(state_maps[-2].file_number, 1)
        self.assertEqual(state_maps[-1].file_number, 1)
        # Test times
        times = db.times()
        self.assertEqual( times[-2], 0.0020000000949949026 )
        self.assertEqual( times[-1], 0.003000000026077032 )

        # Test the results we wrote out.
        vy = db.query("vy", "node", labels=[100,101,102], states=[102,103])
        np.testing.assert_equal(vy['vy']['data'][0,:,:], np.array([[100.0], [100.0], [100.0]]))
        np.testing.assert_equal(vy['vy']['data'][1,:,:], np.array([[200.0], [300.0], [400.0]]))

        sx = db.query("sx", "brick", labels=[1,2,3], states=[102,103])
        np.testing.assert_equal(sx['sx']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(sx['sx']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        strain = db.query("strain", "brick", labels=[10,11,12], states=[102,103])
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]))
        np.testing.assert_equal(strain['strain']['data'][0,2,:], np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]))
        np.testing.assert_equal(strain['strain']['data'][1,2,:], np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0]))

        edrate = db.query("edrate", "shell", labels=[7,8,9], states=[102,103], ips=1)
        np.testing.assert_equal(edrate['edrate']['data'][0,:,:], np.array([[10.0], [20.0], [30.0]]))
        np.testing.assert_equal(edrate['edrate']['data'][1,:,:], np.array([[40.0], [50.0], [60.0]]))

        strain = db.query("strain", "shell", labels=[5,6], states=[102,103], ips=[2])
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([17.0, 18.0, 19.0, 20.0, 21.0, 22.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([17.0, 18.0, 19.0, 20.0, 21.0, 22.0]))

        axf = db.query("axf", "beam", labels=[1,2,3], states=[102,103])
        np.testing.assert_equal(axf['axf']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(axf['axf']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        stress = db.query("stress", "beam", labels=[43], states=[102,103])
        np.testing.assert_equal(stress['stress']['data'][0,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]))
        np.testing.assert_equal(stress['stress']['data'][1,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]))
        os.remove(f"{db_name}01")

    def test_create_new_database(self):
        db_name = f"{self._testMethodName}_{TestAppendStatesToolSerial.base_name}"
        new_db_name = f"{self._testMethodName}_{TestAppendStatesToolSerial.new_base_name}"
        append_states_spec = {
            "database_basename": db_name,
            "output_type": ["mili"],
            "output_mode": "write",
            "output_basename": new_db_name,
            "states": 2,
            "time_inc": 0.001,
            "limit_bytes_per_file": 100,

            "state_variables": {
                "node": {
                    "vy": {
                        'labels': [100, 101, 102],
                        'data': [
                            [100.0, 100.0, 100.0],
                            [200.0, 300.0, 400.0],
                        ]
                    }
                },
                "brick": {
                    # Scalar state variable (no int_points)
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable (no int_points)
                    "strain": {
                        'labels': [10, 11, 12],
                        'data': [
                            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                        ]
                    }
                },
                "shell": {
                    # Scalar state variable w/ SINGLE int_points
                    "edrate": {
                        "labels": [7, 8, 9],
                        'int_point': 1,
                        'data': [
                            [10.0, 20.0, 30.0],
                            [40.0, 50.0, 60.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_point)
                    "strain": {
                        'labels': [5,6],
                        'int_point': 2,
                        'data': [
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
                        ]
                    }
                },
                "beam": {
                    # Scalar state variable (no int_points)
                    "axf": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_points)
                    "stress": {
                        'labels': [43],
                        # Data for 1 element with 4 integration points
                        'data': [
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0],
                        ]
                    }
                },
            }
        }
        tool = AppendStatesTool(append_states_spec)
        tool.write_states()

        # Verify new database looks correct.
        db = reader.open_database( new_db_name )

        # Test state maps
        state_maps = db.state_maps()
        self.assertEqual( len(state_maps), 2 )
        self.assertEqual(state_maps[0].file_number, 0)
        self.assertEqual(state_maps[1].file_number, 1)
        times = db.times()
        self.assertEqual( times[-2], 0.0020000000949949026 )
        self.assertEqual( times[-1], 0.003000000026077032 )

        # Test the results we wrote out.
        vy = db.query("vy", "node", labels=[100,101,102], states=[1,2])
        np.testing.assert_equal(vy['vy']['data'][0,:,:], np.array([[100.0], [100.0], [100.0]]))
        np.testing.assert_equal(vy['vy']['data'][1,:,:], np.array([[200.0], [300.0], [400.0]]))

        sx = db.query("sx", "brick", labels=[1,2,3], states=[1,2])
        np.testing.assert_equal(sx['sx']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(sx['sx']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        strain = db.query("strain", "brick", labels=[10,11,12], states=[1,2])
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]))
        np.testing.assert_equal(strain['strain']['data'][0,2,:], np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]))
        np.testing.assert_equal(strain['strain']['data'][1,2,:], np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0]))

        edrate = db.query("edrate", "shell", labels=[7,8,9], states=[1,2], ips=1)
        np.testing.assert_equal(edrate['edrate']['data'][0,:,:], np.array([[10.0], [20.0], [30.0]]))
        np.testing.assert_equal(edrate['edrate']['data'][1,:,:], np.array([[40.0], [50.0], [60.0]]))

        strain = db.query("strain", "shell", labels=[5,6], states=[1,2], ips=[2])
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([17.0, 18.0, 19.0, 20.0, 21.0, 22.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([17.0, 18.0, 19.0, 20.0, 21.0, 22.0]))

        axf = db.query("axf", "beam", labels=[1,2,3], states=[1,2])
        np.testing.assert_equal(axf['axf']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(axf['axf']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        stress = db.query("stress", "beam", labels=[43], states=[1,2])
        np.testing.assert_equal(stress['stress']['data'][0,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]))
        np.testing.assert_equal(stress['stress']['data'][1,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0]))

class TestAppendStatesToolParallel(unittest.TestCase):
    """Tests the AppendStatesTool object on an uncombined database."""
    data_path = os.path.join(dir_path,'data','parallel','d3samp6')
    required_plot_files = ["d3samp6.plt000A", "d3samp6.plt00000",
                           "d3samp6.plt001A", "d3samp6.plt00100",
                           "d3samp6.plt002A", "d3samp6.plt00200",
                           "d3samp6.plt003A", "d3samp6.plt00300",
                           "d3samp6.plt004A", "d3samp6.plt00400",
                           "d3samp6.plt005A", "d3samp6.plt00500",
                           "d3samp6.plt006A", "d3samp6.plt00600",
                           "d3samp6.plt007A", "d3samp6.plt00700"]
    base_name = 'd3samp6.plt'
    new_base_name = 'copy_d3samp6.plt'

    def copy_required_plot_files(self, path, required_files, test_prefix):
        for file in required_files:
            fname = os.path.join(path, file)
            shutil.copyfile(fname, f"./{test_prefix}_{file}")

    def setUp(self):
        self.copy_required_plot_files(TestAppendStatesToolParallel.data_path,
                                      TestAppendStatesToolParallel.required_plot_files,
                                      self._testMethodName)


    def tearDown(self):
        # Only delete temp files if test passes.
        if self._outcome.success:
            for f in TestAppendStatesToolParallel.required_plot_files:
                os.remove(f"./{self._testMethodName}_{f}")
            for proc in ["000", "001", "002", "003", "004", "005", "006", "007"]:
                if os.path.exists(f"./{self._testMethodName}_{TestAppendStatesToolParallel.new_base_name}{proc}A"):
                    os.remove(f"./{self._testMethodName}_{TestAppendStatesToolParallel.new_base_name}{proc}A")
                if os.path.exists(f"./{self._testMethodName}_{TestAppendStatesToolParallel.new_base_name}{proc}00"):
                    os.remove(f"./{self._testMethodName}_{TestAppendStatesToolParallel.new_base_name}{proc}00")

    def test_append_to_existing(self):
        db_name = f"{self._testMethodName}_{TestAppendStatesToolParallel.base_name}"
        append_states_spec = {
            "database_basename": db_name,
            "output_type": ["mili"],
            "output_mode": "append",
            "states": 2,
            "time_inc": 0.001,

            "state_variables": {
                "node": {
                    "vy": {
                        'labels': [100, 101, 102],
                        'data': [
                            [100.0, 100.0, 100.0],
                            [200.0, 300.0, 400.0],
                        ]
                    }
                },
                "brick": {
                    # Scalar state variable (no int_points)
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable (no int_points)
                    "strain": {
                        'labels': [10, 11, 12],
                        'data': [
                            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                        ]
                    }
                },
                "shell": {
                    # Vector state variable w/ ALL int_point)
                    "es_3c": {
                        'labels': [5,6],
                        'int_point': 2,
                        'data': [
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                        ]
                    }
                },
                "beam": {
                    # Scalar state variable w/ SINGLE int_point
                    "axf": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_points)
                    "es_1a": {
                        'labels': [43],
                        # Data for 1 element with 4 integration points
                        'data': [
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0],
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0],
                        ]
                    }
                },
            }
        }
        tool = AppendStatesTool(append_states_spec)
        tool.write_states()

        # Verify new database looks correct.
        db = reader.open_database( db_name, merge_results=False )

        # Test state maps
        state_maps = db.state_maps()
        self.assertEqual( len(state_maps), 8 )
        for smap in state_maps:
            self.assertEqual( len(smap), 103 )
        times = db.times()
        for proc_times in times:
            self.assertEqual( proc_times[-2], 0.0020000000949949026 )
            self.assertEqual( proc_times[-1], 0.003000000026077032 )

        # Test the results we wrote out.
        vy = reader.combine(db.query("vy", "node", labels=[100,101,102], states=[102,103]))
        np.testing.assert_equal(vy['vy']['data'][0,:,:], np.array([[100.0], [100.0], [100.0]]))
        np.testing.assert_equal(vy['vy']['data'][1,:,:], np.array([[200.0], [300.0], [400.0]]))

        sx = reader.combine(db.query("sx", "brick", labels=[1,2,3], states=[102,103]))
        np.testing.assert_equal(sx['sx']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(sx['sx']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        strain = reader.combine(db.query("strain", "brick", labels=[10,11,12], states=[102,103]))
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]))
        np.testing.assert_equal(strain['strain']['data'][0,2,:], np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]))
        np.testing.assert_equal(strain['strain']['data'][1,2,:], np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0]))

        strain = reader.combine(db.query("es_3c", "shell", labels=[5,6], states=[102,103], ips=[2]))
        np.testing.assert_equal(strain['es_3c']['data'][0,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]))
        np.testing.assert_equal(strain['es_3c']['data'][0,1,:], np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]))
        np.testing.assert_equal(strain['es_3c']['data'][1,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]))
        np.testing.assert_equal(strain['es_3c']['data'][1,1,:], np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]))

        axf = reader.combine(db.query("axf", "beam", labels=[1,2,3], states=[102,103]))
        np.testing.assert_equal(axf['axf']['layout']['labels'], np.array([2, 3, 1]))
        np.testing.assert_equal(axf['axf']['data'][0,:,:], np.array([[2.0], [3.0], [1.0]]))
        np.testing.assert_equal(axf['axf']['data'][1,:,:], np.array([[5.0], [6.0], [4.0]]))

        stress = reader.combine(db.query("es_1a", "beam", labels=[43], states=[102,103]))
        np.testing.assert_equal(stress['es_1a']['data'][0,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]))
        np.testing.assert_equal(stress['es_1a']['data'][1,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]))
        db.close()

    def test_create_new_database(self):
        db_name = f"{self._testMethodName}_{TestAppendStatesToolParallel.base_name}"
        new_db_name = f"{self._testMethodName}_{TestAppendStatesToolParallel.new_base_name}"
        append_states_spec = {
            "database_basename": db_name,
            "output_type": ["mili"],
            "output_mode": "write",
            "output_basename": new_db_name,
            "states": 2,
            "time_inc": 0.001,

            "state_variables": {
                "node": {
                    "vy": {
                        'labels': [100, 101, 102],
                        'data': [
                            [100.0, 100.0, 100.0],
                            [200.0, 300.0, 400.0],
                        ]
                    }
                },
                "brick": {
                    # Scalar state variable (no int_points)
                    "sx": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable (no int_points)
                    "strain": {
                        'labels': [10, 11, 12],
                        'data': [
                            [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0],
                            [40.0, 40.0, 40.0, 40.0, 40.0, 40.0, 50.0, 50.0, 50.0, 50.0, 50.0, 50.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0],
                        ]
                    }
                },
                "shell": {
                    # Vector state variable w/ ALL int_point)
                    "es_3c": {
                        'labels': [5,6],
                        'int_point': 2,
                        'data': [
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                            [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0],
                        ]
                    }
                },
                "beam": {
                    # Scalar state variable w/ SINGLE int_point
                    "axf": {
                        "labels": [1, 2, 3],
                        'data': [
                            [1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                        ]
                    },
                    # Vector state variable w/ ALL int_points)
                    "es_1a": {
                        'labels': [43],
                        # Data for 1 element with 4 integration points
                        'data': [
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0],
                            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0],
                        ]
                    }
                },
            }
        }
        tool = AppendStatesTool(append_states_spec)
        tool.write_states()

        # Verify new database looks correct.
        db = reader.open_database( new_db_name, merge_results=False)

        # Test state maps
        state_maps = db.state_maps()
        self.assertEqual( len(state_maps), 8 )
        for smap in state_maps:
            self.assertEqual( len(smap), 2 )
        times = db.times()
        for proc_times in times:
            self.assertEqual( proc_times[-2], 0.0020000000949949026 )
            self.assertEqual( proc_times[-1], 0.003000000026077032 )

        # Test the results we wrote out.
        vy = reader.combine(db.query("vy", "node", labels=[100,101,102], states=[1,2]))
        np.testing.assert_equal(vy['vy']['data'][0,:,:], np.array([[100.0], [100.0], [100.0]]))
        np.testing.assert_equal(vy['vy']['data'][1,:,:], np.array([[200.0], [300.0], [400.0]]))

        sx = reader.combine(db.query("sx", "brick", labels=[1,2,3], states=[1,2]))
        np.testing.assert_equal(sx['sx']['data'][0,:,:], np.array([[1.0], [2.0], [3.0]]))
        np.testing.assert_equal(sx['sx']['data'][1,:,:], np.array([[4.0], [5.0], [6.0]]))

        strain = reader.combine(db.query("strain", "brick", labels=[10,11,12], states=[1,2]))
        np.testing.assert_equal(strain['strain']['data'][0,0,:], np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0]))
        np.testing.assert_equal(strain['strain']['data'][0,1,:], np.array([20.0, 20.0, 20.0, 20.0, 20.0, 20.0]))
        np.testing.assert_equal(strain['strain']['data'][0,2,:], np.array([30.0, 30.0, 30.0, 30.0, 30.0, 30.0]))
        np.testing.assert_equal(strain['strain']['data'][1,0,:], np.array([40.0, 40.0, 40.0, 40.0, 40.0, 40.0]))
        np.testing.assert_equal(strain['strain']['data'][1,1,:], np.array([50.0, 50.0, 50.0, 50.0, 50.0, 50.0]))
        np.testing.assert_equal(strain['strain']['data'][1,2,:], np.array([60.0, 60.0, 60.0, 60.0, 60.0, 60.0]))

        strain = reader.combine(db.query("es_3c", "shell", labels=[5,6], states=[1,2], ips=[2]))
        np.testing.assert_equal(strain['es_3c']['data'][0,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]))
        np.testing.assert_equal(strain['es_3c']['data'][0,1,:], np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]))
        np.testing.assert_equal(strain['es_3c']['data'][1,0,:], np.array([11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]))
        np.testing.assert_equal(strain['es_3c']['data'][1,1,:], np.array([18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0]))

        axf = reader.combine(db.query("axf", "beam", labels=[1,2,3], states=[1,2]))
        np.testing.assert_equal(axf['axf']['layout']['labels'], np.array([2, 3, 1]))
        np.testing.assert_equal(axf['axf']['data'][0,:,:], np.array([[2.0], [3.0], [1.0]]))
        np.testing.assert_equal(axf['axf']['data'][1,:,:], np.array([[5.0], [6.0], [4.0]]))

        stress = reader.combine(db.query("es_1a", "beam", labels=[43], states=[1,2]))
        np.testing.assert_equal(stress['es_1a']['data'][0,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]))
        np.testing.assert_equal(stress['es_1a']['data'][1,0,:], np.array([1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 6.0, 6.0, 6.0, 6.0, 7.0, 7.0, 7.0, 7.0]))
        db.close()
