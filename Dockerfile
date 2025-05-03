# Use Ubuntu 20.04 as base image
FROM ubuntu:20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y curl gnupg2 lsb-release

# Add the ROS repository and update apt sources
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN apt-get update

# Install all the things
RUN apt-get install -y \
  ros-noetic-desktop-full \
  python3-rosdep \
  python3-rosinstall \
  python3-rosinstall-generator \
  python3-wstool \
  build-essential \
  python3-catkin-tools \
  ros-noetic-rosserial-arduino \
  ros-noetic-rosserial \
  ros-noetic-librealsense2 \
  ros-noetic-realsense2-camera \
  ros-noetic-realsense2-description

RUN rosdep init && rosdep update

RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc

RUN mkdir /catkin_ws
WORKDIR /catkin_ws

SHELL ["/bin/bash", "-c"] 
COPY ./catkin_ws/src /catkin_ws/src
#COPY ./setup.bash /setup.bash

RUN source /opt/ros/noetic/setup.bash && \
  cd /catkin_ws && \
  catkin build

# Update package lists and install necessary dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy and install Python requirements
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Default command to execute when the container starts
CMD [ "python3", "./scripts/main.py" ]
