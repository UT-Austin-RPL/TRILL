![architecture diagram](architecture_diagram.jpeg)

A script for getting controller input from and streaming stereoscopic video to a VR headset, built with OpenVR.

## Introduction

Developed for teleoperation of robots, the script reports the 6-DOF poses of left and right controllers as well as additional buttons for locomotion and gripper control.
In addition, stereoscopic images are streamed and displayed on the VR headset to create depth perception for the wearer.
Using ZMQ, the script can connect to both our simulation written in python and our real-robot controller written in C++.

The protobuf file is contained in the messages folder. Examples on how to interact with the script is contained in the examples folder.

### Architecture

We use OpenVR (implemented in SteamVR) and ALVR (Air Light VR) for the communication between the laptop and the headset. OpenVR is a low-level API designed to support a wide range of VR devices. ALVR is an open-source project that allows streaming Steam VR games from the laptop to the headset via Wi-Fi. It implements technologies such as Asynchronous Timewarp and Fixed Foveated Rendering for a smoother experience. ZMQ is an asynchronous messaging library that simplies message-passing between different programs or devices.

For the simulation, we use Mujoco with Python binding for the physical simulation and Robosuite for the objects in the scene. We first render the scene using a virtual stereoscopic camera that is adjusted to match the interpupillary distance of the VR headset. Then, the rendered pixels are copied from the GPU to CPU and sent to the VR interface using ZMQ and ethernet. The interface code listens to the images and writes them into a GPU texture used by OpenVR. Finally, SteamVR and ALVR transmit the images through a router and displays them in the VR headset. At the same time, the interface polls the VR headset for the poses of the headset and controllers through OpenVR. It transforms the controller poses to the local frame of the headset, and they are then mapped to the poses of the robot hands in the robot’s local frame.
The transformed hand poses are then published continuously using a ZMQ pub socket. When the simulation needs a VR command, it pulls the most recent command from the queue and sends it to the whole-body controller.

## Ubuntu Installation Instructions (with GPU)

### Installing ALVR on Ubuntu

1. Install steam. If this step doesn’t work, reinstall the apt repository
   1. `sudo add-apt-repository multiverse`
   2. `sudo apt update`
   3. `sudo apt install steam`
2. Run steam and install steamVR by steam GUI
3. Launch steamVR and close it
4. Install ALVR version 18.2.3 by downloading the tar file on this page [Release ALVR v18.2.3 · alvr-org/ALVR](https://github.com/alvr-org/ALVR/releases/tag/v18.2.3)

   1. `curl -L -O https://github.com/alvr-org/ALVR/releases/download/v18.2.3/alvr_server_linux_portable.tar.gz`

5. If on step 4 you didn’t install the portable version, then install ffmpeg as required by this page: [https://github.com/alvr-org/ALVR/releases/tag/v18.2.3](https://github.com/alvr-org/ALVR/releases/tag/v18.2.3)
6. Install chrome/chromium
   1. `sudo apt install chromium-browser`
7. launch ALVR

### Installing ALVR on Oculus Quest

Follow “headset side” of this guide:

[Basic installation · alvr-org/ALVR Wiki](https://github.com/alvr-org/ALVR/wiki/Basic-installation#other)

Essentially you have to install side quest to side load the app to the oculus quest. After this, connecting the quest to the same network and launching the alvr app should work.

### Building this codebase on Ubuntu

1. Clone this GitHub repo

2. Install conan. If you want Conan to be able to automatically apt install missing packages, make sure to install Conan globally. Otherwise, install it locally.
   Make sure to install the 1.x version of conan as the current build file isn't compatible with 2.x.
   1. `pip install conan==1.60.2`
3. In the GitHub repo, install dependencies with Conan
   1. `cd vr_interface`
   2. `mkdir build`
   3. `cd build`
   4. `conan profile new default --detect`
   5. `conan profile update settings.compiler.libcxx**=**libstdc++11 default`
   6. if you want to let Conan apt install missing packages, `conan profile update conf.tools.system.package_manager:mode=install default` and `conan install ..`
   7. `conan profile update conf.tools.system.package_manager:sudo=True default`
   8. otherwise, simply run `conan install ..` Note that this might require manually installing missing packages
4. In the same folder, build with CMake
   1. `cmake ..`
   2. `make`
5. ~/.steam/steam/ubuntu12_32/steam-runtime/run.sh ./bin/vr

## VR Usage Instructions

The code is meant to be run on the GPU laptop. ALVR is used to connect the laptop and the VR headset. ALVR uses a desktop client and a VR headset side-loaded app to establish connection over a local network.

To run this setup, you need to first

### Launch the ALVR app on the laptop

1. Navigate to the `~/alvr` folder or wherever alvr is installed
2. Run `./bin/alvr_launcher.` This should automatically start Steam and ALVR, and you’ll be greeted with a dashboard.
3. Turn on the wired connection on the GPU laptop to connect to the router. Note that this will disrupt your internet access since the router doesn’t have wifi connection.

If the ALVR dashboard is already running, you might have to restart it since the previous session might be stale. Also, every time that you quit the VR interface script, you would have to restart the ALVR to avoid a segfault. Here’s how

### Restart ALVR if needed

1. If the dashboard is open, you can click on the Restart SteamVR button. Make sure the wired internet is off, since Steam might complain if the internet access is off (you can still launch it in offline mode, but you might get the error “error launching steamvr: app running”).
   1. If you were to get this error, or if there are any other errors, you can restart the whole steam application. See below for instructions.
2. Once the VR dashboard is opened, you are ready to go.
3. If this way doesn’t work, you’d have to restart the entire program.
   1. Find steam and steamVR on the menu bar. Right click on them and quit them.
   2. Wait for them to quit. Also quit alvr by closing the dashboard window.
   3. Try re-launching alvr. If it doesn’t launch, there might be remaining steam processes that are still running.
      1. Execute `ps -A | grep “steam”` to find them
      2. Execute `kill -9 <pid>` to kill them
      3. Now you should be able to relaunch ALVR.

After ALVR is up and running, you can start up the VR interface script.

### Launch VR Interface Script

In the `vr_interface` directory, execute `~/.steam/steam/ubuntu12_32/steam-runtime/run.sh build/bin/vr`.

Now you’re ready to connect to the headset.

### Launch the ALVR app on the Oculus Quest

1. Open the app drawer
2. On the top right corner, click on the drop-down menu “All”.
3. Scroll down and choose “Unknown Sources”.
4. Select ALVR in the list.

Make sure that the headset and laptop are connected to the same wifi. You should now see a screen with mountains (before executing the VR interface script, you would see a SteamVR room).

## Troubleshooting

1. Don't run 2 versions of the vr script (for example, if you modify the script, put it in a separate repo, and run the modified script before running the original script). When this happens, SteamVR tends to work for a few seconds before crashing. The solution is to restart the laptop and make sure to only run one version of the script.

## Design Choices

Some design decisions were made to satisfy the requirements defined above. They are explained in this section in hopes of providing documentation and guidance for others working on similar systems.

### OpenVR

Initially, a website was created based on WebXR and JavaScript to stream controller poses over the internet. I wanted to improve the latency by keeping the connection within a local network, but I encountered difficulty in setting up HTTPS certificates in the campus lab. After getting tired of tunneling the connection over the internet, I decided to pursue a more low-level approach.
The native Oculus SDK could work, but it would limit the option to change VR systems in the future. Unity is a good option for cross-platform compatibility, but since there's only a need to stream images and controller poses, a game engine is an overkill. I also don't believe that Unity exposes the low-level functionality to directly show the stereoscopic images in the headset. Since Unity uses the low-level OpenVR API, I decided to use it directly. Although a newer API called OpenXR is gaining steam, I feel that the performance benefits of OpenXR doesn't justify its complexity compared to OpenVR.

Using OpenVR has several advantages. First, it has great performance since it is used by VR games and has direct support from VR headset manufacturers. Second, it delegates the work of video streaming to existing technologies. Many VR headsets support direct HDMI connection from the computer's graphics card, which OpenVR can take full advantage of since it takes input images from OpenGL. Even though the Quest doesn't have an HDMI cable, it is still possible to use the Oculus Link (over an USB cable) or Oculus Air Link (over a local network) with OpenVR. These are sophisticated streaming technologies that predict the user's movements and streaming latency to render ahead of time, and they encode the frames as slices in H.264. ALVR is an open-source alternative that implements similar technologies with the added benefit of having experimental support for Linux. Using these technologies is much more performant and scalable than hand-coding an image streaming pipeline. Finally, OpenVR works on a variety of Operating Systems and targets many VR headsets.

### Image Transfer from GPU to CPU

Since our interface script is written in C++ (since the Python OpenVR binding doesn't work well) and the simulation is in Python, we need a way to transfer the rendered images between processes.
First, we can use memory sharing or pybind to transfer the Mujoco simulation state. Using the simulation states, the C++ code can directly render the scene using Mujoco and pass the result directly to OpenVR within the GPU. However, since Mujoco allocates simulation data structure dynamically, it's difficult to do inter-process memory sharing. Second, we could hypothetically share the GPU buffer rendered by the python script with the C++ script, but OpenGL contexts don't seem to allow inter-process sharing. Third, we can lose some performance and transfer the images from GPU to CPU. Once the images are in memory, we can send them over using ZMQ.

The last approach was chosen to make the design more adaptable to the real robot, where the images are always coming from memory. The rendered images have a resolution of $1096 \times 2$ by $1176$, where the width is multiplied by 2 to account for both eyes. The rendering of an image takes .3 milliseconds on the RTX3090 GPU, copying it to the CPU consumes 3 milliseconds, and sending it through ZMQ asynchronously uses .6 milliseconds. Overall, this number is negligible compared to the simulation and whole-body control times, so this performance is acceptable.

### Asynchronous Message Passing

The simulation uses a ZMQ sub socket to get actions from the pub socket in the interface script. Usually this is done synchronously, meaning that the simulation waits for the interface to provide a response after requesting. However, this approach usually takes 70 milliseconds for the message round-trip. This delay can be reduced to .1 milliseconds by using an asynchronous approach. In this case, the sender and receiver simply run at their own pace, and the ZMQ threads in the background takes care of getting the messages ready. When the simulation requests an action, the ZMQ simply gets the most recently received message in its queue and returns it. Since the interface runs at a higher frequency than the simulation, there will never be a case where no action is available. Also, by setting the "conflate" option in ZMQ, we can reduce the need of a queue by only keeping the most recent message.

The performance of the VR Interface is acceptable. Using asynchronous message passing and a separate laptop for VR interface, receiving images and sending commands are both sub-millisecond operations. The biggest contributor to latency is ALVR, which adds about 70 milliseconds of latency.
