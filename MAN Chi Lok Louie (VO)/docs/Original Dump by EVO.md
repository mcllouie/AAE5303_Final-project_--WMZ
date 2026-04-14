================================================================================
AAE5303 MONOCULAR VO EVALUATION (evo)
================================================================================
Ground truth: AMtown02_ground_truth.txt
Estimated:    AMtown02-4_ALL.txt
Association:  t_max_diff = 0.100 s
RPE delta:    10.000 m

--------------------------------------------------------------------------------
Loaded 3750 stamps and poses from: AMtown02_ground_truth.txt
Loaded 7052 stamps and poses from: AMtown02-4_ALL.txt
--------------------------------------------------------------------------------
Synchronizing trajectories...
Found 3535 of max. 3750 possible matching timestamps between...
	AMtown02_ground_truth.txt
and:	AMtown02-4_ALL.txt
..with max. time diff.: 0.1 (s) and time offset: 0.0 (s).
--------------------------------------------------------------------------------
Aligning using Umeyama's method... (with scale correction)
Rotation of alignment:
[[ 0.60870418 -0.79329022 -0.01303234]
 [-0.79315934 -0.6080373  -0.03448046]
 [ 0.01942887  0.03132512 -0.9993204 ]]
Translation of alignment:
[-2.95181972 -0.41600864 15.3305147 ]
Scale correction: 14.304133685260755
--------------------------------------------------------------------------------
Compared 3535 absolute pose pairs.
Calculating APE for translation part pose relation...
--------------------------------------------------------------------------------
APE w.r.t. translation part (m)
(with Sim(3) Umeyama alignment)

       max	12.521220
      mean	2.640429
    median	2.452567
       min	0.419808
      rmse	2.906494
       sse	29862.646034
       std	1.214843

--------------------------------------------------------------------------------
Saving results to evaluation_results/ate.zip...
--------------------------------------------------------------------------------
Loaded 3750 stamps and poses from: AMtown02_ground_truth.txt
Loaded 7052 stamps and poses from: AMtown02-4_ALL.txt
--------------------------------------------------------------------------------
Synchronizing trajectories...
Found 3535 of max. 3750 possible matching timestamps between...
	AMtown02_ground_truth.txt
and:	AMtown02-4_ALL.txt
..with max. time diff.: 0.1 (s) and time offset: 0.0 (s).
--------------------------------------------------------------------------------
Aligning using Umeyama's method... (with scale correction)
Rotation of alignment:
[[ 0.60870418 -0.79329022 -0.01303234]
 [-0.79315934 -0.6080373  -0.03448046]
 [ 0.01942887  0.03132512 -0.9993204 ]]
Translation of alignment:
[-2.95181972 -0.41600864 15.3305147 ]
Scale correction: 14.304133685260755
--------------------------------------------------------------------------------
Found 470 pairs with delta 10.0 (m) among 3535 poses using consecutive pairs.
Compared 470 relative pose pairs, delta = 10.0 (m) with consecutive pairs.
Calculating RPE for translation part pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. translation part (m)
for delta = 10.0 (m) using consecutive pairs
(with Sim(3) Umeyama alignment)

       max	23.222603
      mean	12.820904
    median	13.025792
       min	0.460566
      rmse	15.056882
       sse	106553.552179
       std	7.895194

--------------------------------------------------------------------------------
Saving results to evaluation_results/rpe_trans.zip...
--------------------------------------------------------------------------------
Loaded 3750 stamps and poses from: AMtown02_ground_truth.txt
Loaded 7052 stamps and poses from: AMtown02-4_ALL.txt
--------------------------------------------------------------------------------
Synchronizing trajectories...
Found 3535 of max. 3750 possible matching timestamps between...
	AMtown02_ground_truth.txt
and:	AMtown02-4_ALL.txt
..with max. time diff.: 0.1 (s) and time offset: 0.0 (s).
--------------------------------------------------------------------------------
Aligning using Umeyama's method... (with scale correction)
Rotation of alignment:
[[ 0.60870418 -0.79329022 -0.01303234]
 [-0.79315934 -0.6080373  -0.03448046]
 [ 0.01942887  0.03132512 -0.9993204 ]]
Translation of alignment:
[-2.95181972 -0.41600864 15.3305147 ]
Scale correction: 14.304133685260755
--------------------------------------------------------------------------------
Found 470 pairs with delta 10.0 (m) among 3535 poses using consecutive pairs.
Compared 470 relative pose pairs, delta = 10.0 (m) with consecutive pairs.
Calculating RPE for rotation angle in degrees pose relation...
--------------------------------------------------------------------------------
RPE w.r.t. rotation angle in degrees (deg)
for delta = 10.0 (m) using consecutive pairs
(with Sim(3) Umeyama alignment)

       max	156.387510
      mean	4.304173
    median	1.078427
       min	0.053874
      rmse	12.127392
       sse	69124.606606
       std	11.337889

--------------------------------------------------------------------------------
Saving results to evaluation_results/rpe_rot.zip...

================================================================================
PARALLEL METRICS (NO WEIGHTING)
================================================================================
ATE RMSE (m):                 2.906494
RPE trans drift (m/m):        1.282090
RPE rot drift (deg/100m):     43.041726
Completeness (%):             94.27  (3535 / 3750)