To Run:

roscore

rosrun image_transport republish compressed in:=/left_camera/image out:=/camera/image_raw

cd ~/ORB_SLAM3
./Examples/ROS/ORB_SLAM3/Mono Vocabulary/ORBvoc.txt Examples/Monocular-Inertial/AAE5303/XXX

cd ~/Downloads/ROS_Bags/
rosbag play --rate 0.5 XXX.bag /dji_osdk_ros/imu:=/imu


Evaluation:

Extract RTK ground truth (Edit Script for Name)

python3 AA_Extract_Ground_Truth.py HKisland_GNSS03.bag --output ground_truth.txt

Evaluate trajectory:

python3 scripts/evaluate_vo_accuracy.py --groundtruth ground_truth.txt \
    --estimated CameraTrajectory.txt \
    --t-max-diff 0.1 --delta-m 10 --workdir evaluation_results --json-out evaluation_results/metrics.json

evo_ape tum groundtruth.txt estimated.txt -a -s --t_max_diff 0.1 --save_results results.zip

python3 scripts/generate_report_figures.py --gt ground_truth.txt \
    --est CameraTrajectory.txt \
    --evo-ape-zip EVO_APE_ZIP \
    --out OUT 

