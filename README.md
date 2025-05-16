# Human Tracking Robot: Perception Stack

This repository provides the perception module for a human-tracking robot. It employs deep learning-based techniques for detecting and re-identifying individuals in the environment.

## Requirements

To run this codebase, the following dependencies must be installed:

- PyTorch  
- OpenCV  
- NumPy  

## Setup Instructions

1. Create a virtual environment in the root folder:

   ```bash
   pip install virtualenv
   virtualenv <env_name>
   ```

2. Activate the environment:

   ```bash
   source <env_name>/bin/activate
   ```

3. Install all necessary packages:

   ```bash
   pip install -r requirements.txt
   ```

## How to Use

1. Begin by setting up SSH communication between your local machine and the Triton robot. Follow the instructions in this [documentation](https://gitlab.com/HCRLab/stingray-robotics/triton_docs).

2. On the Triton robot, launch the `stingray_camera` package:

   ```bash
   cd ~/catkin_ws
   source devel/setup.bash
   roslaunch stingray_camera triton.launch
   ```

3. Then, on your local machine, run the perception system:

   ```bash
   python3 scripts/main.py
   ```

## Algorithms Integrated

This stack includes:

- **YOLOv7** for detecting and segmenting humans  
- **SuperPoint** for extracting and matching visual features  

## Performance

The system reliably identifies and tracks a specific person, enabling the robot to follow them effectively.

## References

This work incorporates methods and code from the following projects and publications:

- [SuperPoint: Self-Supervised Interest Point Detection and Description](https://arxiv.org/abs/1712.07629)  
- [YOLOv7 Repository](https://github.com/WongKinYiu/yolov7/tree/mask)  
- [Magic Leap’s SuperPoint Network](https://github.com/magicleap/SuperPointPretrainedNetwork)
