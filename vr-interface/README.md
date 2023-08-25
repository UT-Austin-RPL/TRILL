![architecture diagram](architecture_diagram.jpeg)

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
   1. `pip install conan`
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
