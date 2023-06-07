# AI4UAV-code
This project is a code simulator designed to provide a realistic and interactive environment for practicing and experimenting with python. The algorithms were initially integrated, using a simulation that control, navigate and run the drone in PX4 version 1.13.1 and MAVSDK software with Python mavsdk library in the Loop (SITL) simulation. Gazebo 11 simulator interface helped us to preview the mission of the UAV.

# Features
## PX4 Autopilot Integration:
The code seamlessly integrates with the PX4 autopilot system, allowing you to leverage its advanced features and capabilities. You can develop and test your UAV control algorithms using the same software stack that runs on real UAVs.
## Gazebo Simulation Environment:
The code utilizes the Gazebo simulator, a widely adopted open-source 3D robotics simulator, to provide a highly realistic and customizable environment. You can create dynamic scenarios, simulate different weather conditions, and test your UAV in various challenging situations.
## Realistic Physics and Sensor Models:
The code incorporates accurate physics models for UAV dynamics, including flight dynamics, aerodynamics, and sensor characteristics. This enables you to simulate realistic UAV behavior and sensor outputs, enhancing the fidelity of your simulations.
## Flexible Configuration and Customization:
The code allows you to configure and customize various aspects of your UAV and simulation environment. You can modify parameters such as UAV properties, sensor models, and environmental conditions to match your specific requirements and experiment with different scenarios.
## ROS Integration: 
The code integrates with the Robot Operating System (ROS), a flexible framework for developing robotic systems. This integration enables seamless communication between your simulated UAV and other ROS-based components, such as perception algorithms or mission planners.
Real-Time Execution: The simulator provides real-time execution of code, allowing you to see the output and any errors instantly. This feature enables you to understand the behavior of your code and debug it efficiently.

# Required dependencies
1. Following the link to set up a PX4 development environment https://docs.px4.io/main/en/dev_setup/dev_env_linux_ubuntu.html
2. Install Python version 3.8


# Getting Started
To get started with the code simulator, follow these steps:
1. Clone the repository to your local machine.
2. Install the required dependencies as specified in the README file.
3. Copy the folder "animated_person" to this directory PX4-Autopilot->Tools->sitl_gazebo->models
4. Copy the file "baylands.world" to this directory PX4-Autopilot->Tools->sitl_gazebo->world
5. Launch the simulator by running the appropriate command "make px4_sitl gazebo_typhoon_h480__baylands -j8".
6. Start coding and experimenting with different code snippets.
7. View the output and debug your code in real-time.
8. Explore the code snippet library for reference and inspiration.

Feel free to contribute to the project by submitting bug reports, feature requests, or pull requests. Your feedback and contributions are greatly appreciated!

# License
This project is licensed under the MIT License. See the LICENSE file for more information.

# Acknowledgments

We would like to thank the open-source community for their contributions and support. Their dedication and passion for coding have greatly influenced the development of this code simulator.

If you have any questions or need assistance, please don't hesitate to reach out. Happy coding!
