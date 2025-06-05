# CSCI576_Project

# Large-Scale Ultra-High-Resolution Images from Videos
This repository has different branches to implement different methods for panorama-stitching: single-pass, batch, sequential, and individual stitching + merging) 
The individual stitching + merging code is in the main branch which gave the best result for the particular given data. For more information on the methods used refer to the power point presentation

# Overview of Final Implementation 
Implemented adaptive frame sampling based on motion heuristics to choose keyframes 
Applied ORB + SIFT feature extraction and RANSAC homography estimation to align frames under noise 
Added fault-tolerant logic to reuse the last valid homography when a frame failed quality checks, avoiding drift, and memory overflow 
Switched to Gaussian-weighted blending, achieving 40% smoother transitions in overlapping regions 
