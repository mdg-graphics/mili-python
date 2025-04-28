#!/usr/bin/env python3
"""
Testing for the plotting module.

SPDX-License-Identifier: (MIT)
"""

import os
import unittest
from mili.reader import open_database
from mili.plotting import MatPlotLibPlotter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))

def areImagesEqual(image1_path, image2_path):
    if not os.path.isfile(image1_path) or not os.path.isfile(image2_path):
        return False

    test_passed = True
    with Image.open(image1_path) as image1, Image.open(image2_path) as image2:
        if image1.size != image2.size:
            return False

        # Convert images to grayscale and get the pixels
        im1_data = np.array( image1.convert("L").getdata() )
        im2_data = np.array( image2.convert("L").getdata() )

        # Convert grayscale pixels to [0.0, 1.0]
        im1_data = im1_data / 255.0
        im2_data = im2_data / 255.0

        # Check all values are close
        test_passed = np.allclose(im1_data, im2_data, rtol=0.03)

    return test_passed


class SharedMatPlotLibTests:
    class MatPlotLibPlotting(unittest.TestCase):
        """Test plotting Mili-python result using MatPlotLibView."""

        #==============================================================================
        def test_plot_single_scalar(self):
            """Test plotting single scalar state variable."""
            baseline = os.path.join(dir_path,'data','image_baselines',f'{self._testMethodName}.baseline.png')
            result = self.mili.query("sx", "brick", labels=[10,11,12])

            plot: MatPlotLibPlotter = MatPlotLibPlotter()
            fig, axs = plot.initialize_plot()
            plot.update_plot(result, axs)

            current = f"{self._testMethodName}.current.png"
            plt.savefig(current)

            passed = areImagesEqual(current, baseline)
            if passed:
                os.remove(current)
            self.assertTrue(passed)

        #==============================================================================
        def test_plot_single_vector(self):
            """Test plotting single vector state variable."""
            baseline = os.path.join(dir_path,'data','image_baselines',f'{self._testMethodName}.baseline.png')
            result = self.mili.query("nodvel", "node", labels=[44,98])

            plot: MatPlotLibPlotter = MatPlotLibPlotter()
            fig, axs = plot.initialize_plot()
            plot.update_plot(result, axs)

            current = f"{self._testMethodName}.current.png"
            plt.savefig(current)

            passed = areImagesEqual(current, baseline)
            if passed:
                os.remove(current)
            self.assertTrue(passed)

        #==============================================================================
        def test_plot_single_vector_array(self):
            """Test plotting single vector array state variable."""
            baseline = os.path.join(dir_path,'data','image_baselines',f'{self._testMethodName}.baseline.png')
            result = self.mili.query("sx", "beam", labels=[12,14])

            plot: MatPlotLibPlotter = MatPlotLibPlotter()
            fig, axs = plot.initialize_plot()
            plot.update_plot(result, axs)

            current = f"{self._testMethodName}.current.png"
            plt.savefig(current)

            passed = areImagesEqual(current, baseline)
            if passed:
                os.remove(current)
            self.assertTrue(passed)

        #==============================================================================
        def test_multiple_results(self):
            """Test plotting multiple results on single Axes."""
            baseline = os.path.join(dir_path,'data','image_baselines',f'{self._testMethodName}.baseline.png')
            result1 = self.mili.query("sx", "brick", labels=[10,11,12])
            result2 = self.mili.query("sz", "brick", labels=[12,13,14])

            plot: MatPlotLibPlotter = MatPlotLibPlotter()
            fig, axs = plot.initialize_plot()
            plot.update_plot(result1, axs)
            plot.update_plot(result2, axs)

            current = f"{self._testMethodName}.current.png"
            plt.savefig(current)

            passed = areImagesEqual(current, baseline)
            if passed:
                os.remove(current)
            self.assertTrue(passed)

        #==============================================================================
        def test_multiple_plots(self):
            """Test plotting with multiple Axes."""
            baseline = os.path.join(dir_path,'data','image_baselines',f'{self._testMethodName}.baseline.png')
            result1 = self.mili.query("sx", "brick", labels=[10,11,12])
            result2 = self.mili.query("vz", "node", labels=[44, 68, 112])

            plot: MatPlotLibPlotter = MatPlotLibPlotter()
            fig, axs = plot.initialize_plot(2,1)
            plot.update_plot(result1, axs[0])
            plot.update_plot(result2, axs[1])

            current = f"{self._testMethodName}.current.png"
            plt.savefig(current)

            passed = areImagesEqual(current, baseline)
            if passed:
                os.remove(current)
            self.assertTrue(passed)


class ParallelWithMergeMatPlotLibTests(SharedMatPlotLibTests.MatPlotLibPlotting):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = open_database( ParallelWithMergeMatPlotLibTests.file_name, suppress_parallel=False, merge_results=True )

    def tearDown(self):
        self.mili.close()

class ParallelNoMergeMatPlotLibTests(SharedMatPlotLibTests.MatPlotLibPlotting):
    file_name = os.path.join(dir_path,'data','parallel','d3samp6','d3samp6.plt')

    def setUp(self):
        self.mili = open_database( ParallelWithMergeMatPlotLibTests.file_name, suppress_parallel=False, merge_results=False )

    def tearDown(self):
        self.mili.close()