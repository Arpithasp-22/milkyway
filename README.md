
Astronomical Image Processing for Star Tracking

Overview

This project implements a classical image-processing pipeline inspired by star tracker systems used in spacecraft Attitude Determination and Control Systems (ADCS). The goal is to detect star-like objects and moving streak-like objects in astronomical images, estimate star centroids with sub-pixel accuracy, and analyze detection behavior under noisy, real-sky–like conditions.

The project emphasizes explainable, deterministic algorithms, which are commonly preferred in aerospace imaging systems.

Key Features

Synthetic generation of astronomical images with:
    Star-like blobs (Gaussian PSF)
    Moving object streaks (linear motion)
    Configurable noise and signal conditions
    Classical blob detection using thresholding and connected components
    Sub-pixel centroid estimation using 2D Gaussian fitting

Visualization of:
    Ground-truth star positions
    Pixel-level centroids
    Sub-pixel Gaussian centroids
    Analysis of Gaussian fit parameters to study differences between stars and streaks


Methodology

Data Generation
Stars are modeled as point sources blurred by a Gaussian PSF.
Moving objects are simulated as linear streaks with varying orientation and length.
Additive Gaussian noise is used to emulate real-sky conditions.

Star Detection
Thresholding followed by connected-component labeling identifies candidate blobs.
Pixel-level centroids are computed for each detected region.
Sub-pixel Localization
A 2D Gaussian is fitted to local image patches around detected blobs.
The fitted Gaussian mean provides a sub-pixel centroid estimate.

Analysis
Sub-pixel localization improves accuracy over pixel centroids for star-like objects.
Gaussian fit parameters (σx, σy) reveal anisotropy for streak-like objects, highlighting model limitations and potential classification cues.

Tools & Libraries
Python
NumPy, SciPy
OpenCV
Matplotlib


Notes
This project focuses on correctness, interpretability, and clarity rather than end-to-end deep learning. It reflects an engineering-first approach suitable for safety-critical and resource-constrained systems.

