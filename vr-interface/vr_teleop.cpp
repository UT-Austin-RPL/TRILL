/* 
 * Program to connect to a VR headset, stream the VR controller inputs via an exposed ZMQ pub socket, and potentially
 * receive images to display in the headset via an exposed ZMQ sub socket.
 *
 * The VR controller inputs are streamed as a protobuf message of type draco::vr_teleop_msg.
 *
 * OpenVR code copied from https://gist.github.com/machinaut/5bebd312d21a66419b1b329e369b6625
 */

#include <chrono>
#include <google/protobuf/stubs/common.h>
#include <signal.h>
#include "openvr.h"
#include <memory>
#include <string>
#include <iostream>
#include <thread>
#include <zmq.h>
#include <zmq.hpp>
#include <string>
#include "GL/glew.h"
#include "GLFW/glfw3.h"
#include "messages/draco.pb.h"
#include <Eigen/Dense>

#define STREAMING false

using namespace vr;
using namespace std;

using Eigen::Matrix3f;
using Eigen::Vector3f;

string STREAMING_SOCKET_ENDPOINT;
string CONTROL_SOCKET_ENDPOINT;

bool DEBUG = false;

enum {
    vBUTTON_TRIGGER = 0, 
    vBUTTON_SIDE,
    vBUTTON_MENU,
    vBUTTON_PAD,

    vNBUTTON
};

struct controller_t {
    int id;
    int idtrigger;
    int idpad;

    bool touch[vNBUTTON];
    bool hold[vNBUTTON];

    float triggerpos;
    float padpos[2];
    float oldtriggerpos;
    float oldpadpos[2];

    float roompos[3];
    float roommat[9];
};

struct HMD_t {
    IVRSystem *system;
    int id;
    unsigned int idtex;
    uint32_t width, height;

    float roompos[3];
    float roommat[9];
};

HMD_t hmd;
controller_t ctl[2];

// the transformed position and orientation of the controllers
// by default, no transformation is applied
Vector3f transformed_left_pos;
Vector3f transformed_right_pos;
Matrix3f transformed_left_ori;
Matrix3f transformed_right_ori;

// create an invisible glfw window as context (if streaming), 
// initialize openvr compositor, and create VR data structures
void v_initPre(void) {
#if STREAMING
    if (!glfwInit()) {
        cout << "error initializing GLFW" << endl;
    }
    // We don't use this window - it's just here for the opengl context
    GLFWwindow* window = glfwCreateWindow(1, 1, "invisible window", NULL, NULL);
    glfwMakeContextCurrent(window);
    if( glewInit()!=GLEW_OK )
        cout << "error initializing glew" << endl;
#endif

    int n, i;
    EVRInitError err = VRInitError_None;
    hmd.system = VR_Init(&err, vr::VRApplication_Scene);
    cout << "finished vr init" << endl;
    if (err != VRInitError_None)
        cout << "Error with init VR runtime: " << VR_GetVRInitErrorAsEnglishDescription(err);

    if(!VRCompositor()) {
        VR_Shutdown();
        cout << "Could not init Compositor";
    }
    VRCompositor()->SetTrackingSpace(TrackingUniverseStanding);

    int cnt = 0;
    hmd.id = -1;
    ctl[0].id = -1;
    ctl[1].id = -1;
    for (n = 0; n < k_unMaxTrackedDeviceCount; ++n) {
        ETrackedDeviceClass cls = hmd.system->GetTrackedDeviceClass(n);

        if (cls == TrackedDeviceClass_HMD) 
            hmd.id = n;

        else if (cls == vr::TrackedDeviceClass_Controller && cnt < 2) {
            ctl[cnt].id = n;
            cnt++;
        }
    }

    if (hmd.id < 0 || ctl[0].id < 0) {
        cout << "Need at least one controller";
    }

    for (n = 0; n < 9; n++) {
        hmd.roommat[n] = 0;
        if (n < 3)
            hmd.roompos[n] = 0;
    }
    hmd.roommat[0] = 1;
    hmd.roommat[4] = 1;
    hmd.roommat[8] = 1;

    // init controller data
    for (n = 0; n < 2; n++) {
        if (ctl[n].id >= 0) {
            ctl[n].idtrigger = -1;
            ctl[n].idpad = -1;
            for (i = 0; i < k_unControllerStateAxisCount; i++) {
                int prop = hmd.system -> GetInt32TrackedDeviceProperty(ctl[n].id, 
                        (ETrackedDeviceProperty)(Prop_Axis0Type_Int32 + i));
                if (prop == k_eControllerAxis_Trigger)
                    ctl[n].idtrigger = i;
                else if( prop==k_eControllerAxis_TrackPad )
                    ctl[n].idpad = i;
            }

            if (ctl[n].idtrigger < 0)
                cout << "ERROR: trigger axis not found" << endl;
        }
    }

    hmd.system -> GetRecommendedRenderTargetSize(&hmd.width, &hmd.height);
    cout << hmd.width << ", " << hmd.height << endl;

#if STREAMING
    // Set up white texture for initialization
    // This texture will be replaced by the actual images to stream to the headset
    glActiveTexture(GL_TEXTURE2);
    glGenTextures(1, &hmd.idtex);
    glBindTexture(GL_TEXTURE_2D, hmd.idtex);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    int size = 2 * hmd.width*hmd.height * 3; // RGB is 3 values. VR has twice the width.
    unsigned char * white = new unsigned char[size];
    for (int i = 0; i < size; ++i) {
        white[i] = 255;
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2*hmd.width, hmd.height, 0, GL_RGB, GL_UNSIGNED_BYTE, white);
    delete[] white;
#endif
}

void v_copyPose(const TrackedDevicePose_t* pose, float* roompos, float* roommat)
{
    if (!pose->bPoseIsValid)
        return;

    const HmdMatrix34_t* p = &pose->mDeviceToAbsoluteTracking;

    roompos[0] = p->m[0][3];
    roompos[1] = p->m[1][3];
    roompos[2] = p->m[2][3];
    roommat[0] = p->m[0][0];
    roommat[1] = p->m[0][1];
    roommat[2] = p->m[0][2];
    roommat[3] = p->m[1][0];
    roommat[4] = p->m[1][1];
    roommat[5] = p->m[1][2];
    roommat[6] = p->m[2][0];
    roommat[7] = p->m[2][1];
    roommat[8] = p->m[2][2];
}

// gets the controller states and also syncs the new image to be displayed in the headset
void v_update(void) {
    int n;

    TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
    VRCompositor()->WaitGetPoses(poses, k_unMaxTrackedDeviceCount, NULL, 0);

    v_copyPose(poses + hmd.id, hmd.roompos, hmd.roommat);

    for (n = 0; n < 2; n++) {
        if (ctl[n].id >= 0) {
            v_copyPose(poses+ctl[n].id, ctl[n].roompos, ctl[n].roommat);
        }

        VRControllerState_t state;
        hmd.system->GetControllerState(ctl[n].id, &state, sizeof(VRControllerState_t));
        ctl[n].triggerpos = state.rAxis[ctl[n].idtrigger].x;
        ctl[n].padpos[0] = state.rAxis[ctl[n].idpad].x;
        ctl[n].padpos[1] = state.rAxis[ctl[n].idpad].y;
    }

    VREvent_t evt;
    while( hmd.system->PollNextEvent(&evt, sizeof(VREvent_t)) )
        if( evt.eventType>=200 && evt.eventType<=203 )
        {
            // get controller
            if( ctl[0].id==evt.trackedDeviceIndex )
                n = 0;
            else if( ctl[1].id==evt.trackedDeviceIndex )
                n = 1;
            else
                continue;
            
            // get button
            int button = vNBUTTON;
            switch( evt.data.controller.button )
            {
            case k_EButton_ApplicationMenu:
                button = vBUTTON_MENU;
                break;

            case k_EButton_Grip:
                button = vBUTTON_SIDE;
                break;

            case k_EButton_SteamVR_Trigger:
                button = vBUTTON_TRIGGER;
                break;

            case k_EButton_SteamVR_Touchpad:
                button = vBUTTON_PAD;
                break;
            }

            // process event.
            switch( evt.eventType )
            {
            case VREvent_ButtonPress:
                ctl[n].hold[button] = true;

                if (DEBUG){
                    if( button==vBUTTON_TRIGGER )
                    {
                        cout << "trigger" << n << endl;
                    }

                    else if( button==vBUTTON_MENU )
                    {
                        cout << "menu" << n << endl;
                    }

                    else if( button==vBUTTON_PAD)
                    {
                        cout << "PAD" << n << " " << ctl[n].padpos[0] << " " << ctl[n].padpos[1] << endl;
                    }

                    else if( button==vBUTTON_SIDE )
                    {
                        cout << "SIDE" << n << endl;
                    }
                }

                break;

            case VREvent_ButtonUnpress:
                ctl[n].hold[button] = false;
                break;

            case VREvent_ButtonTouch:
                ctl[n].touch[button] = true;

                // reset old axis pos
                if( button==vBUTTON_TRIGGER )
                    ctl[n].oldtriggerpos = ctl[n].triggerpos;
                else if ( button==vBUTTON_PAD )
                {
                    ctl[n].oldpadpos[0] = ctl[n].padpos[0];
                    ctl[n].oldpadpos[1] = ctl[n].padpos[1];
                }
                break;

            case VREvent_ButtonUntouch:
                ctl[n].touch[button] = false;
                break;
            }
        }
}

// submit texture to vr
void v_render(void) {
    glActiveTexture(GL_TEXTURE2);
    const VRTextureBounds_t boundLeft = {0, 0, 0.5, 1};
    const VRTextureBounds_t boundRight = {0.5, 0, 1, 1};
    Texture_t vTex = {(void*)hmd.idtex, TextureType_OpenGL, ColorSpace_Gamma};
    VRCompositor()->Submit(Eye_Left, &vTex, &boundLeft);
    VRCompositor()->Submit(Eye_Right, &vTex, &boundRight);
}

void quit_handler(int s){
    if (s == 2) {
        glDeleteTextures(1, &hmd.idtex);
        exit(1);
    }
}

// If your robot environment has the same coordinate system as the VR controllers, 
// then no transformation is needed.

void transformVRPose() {

}

// This is our code for transforming the VR coordinate system to the 
// coordinate system used for the robot.
// It essentially implements the following equation: 
// Let R_vr be the rotation of the controller from its natural orientation,
// R_t be the transformation between the VR axes and the simulation axes,
// and R_sim be the desired rotation to be applied to the robot hands.
// Then, R_sim = R_t^-1  R_vr  R_t

/*
Matrix3f convert_orientation(Matrix3f VR_orientation){
  Matrix3f T;
  T << 0, -1, 0,
     -1, 0, 0,
     0, 0, -1;
  Matrix3f B_inv;
  B_inv << 1, 0, 0,
       0, 0, 1,
       0, -1, 0;
  Matrix3f rpc_rearrange_axes;
  return T*B_inv*VR_orientation* T.transpose() ;
}

void transformVRPose(){
    Vector3f hmd_pos;
    Matrix3f hmd_mat;
    Vector3f left_pos;
    Matrix3f left_mat;
    Vector3f right_pos;
    Matrix3f right_mat;

    for(int i=0; i<3; i++){
    	hmd_pos(i) = hmd.roompos[i];
        left_pos(i) = ctl[0].roompos[i];
        right_pos(i) = ctl[1].roompos[i];
    }

    for(int i=0; i<3; i++){
        for(int j=0; j<3; j++){
            hmd_mat(i,j) = hmd.roommat[i*3 + j];
            left_mat(i,j) = ctl[0].roommat[i*3 + j];
            right_mat(i,j) = ctl[1].roommat[i*3 + j];
        }
    }
    
    Vector3f local_left_pos = left_pos - hmd_pos;
    Vector3f local_right_pos = right_pos - hmd_pos;
    Matrix3f mat_room2hmd = hmd_mat.inverse();

    local_left_pos = mat_room2hmd * local_left_pos; 
    local_right_pos = mat_room2hmd * local_right_pos;
    
    transformed_left_pos = -1 * local_left_pos;
    float tempValue = transformed_left_pos(2);
    transformed_left_pos(2) = transformed_left_pos(1);
    transformed_left_pos(1) = transformed_left_pos(0);
    transformed_left_pos(0) = tempValue;

    transformed_left_pos(2) = -transformed_left_pos(2) + .2;
    transformed_left_pos(0) = transformed_left_pos(0) + 0.15;

    transformed_right_pos = -1 * local_right_pos;
    tempValue = transformed_right_pos(2);
    transformed_right_pos(2) = transformed_right_pos(1);
    transformed_right_pos(1) = transformed_right_pos(0);
    transformed_right_pos(0) = tempValue;

    transformed_right_pos(2) = -transformed_right_pos(2) + .2;
    transformed_right_pos(0) += .15;

    transformed_left_ori = convert_orientation(mat_room2hmd * left_mat);
    transformed_right_ori = convert_orientation(mat_room2hmd * right_mat);
}
*/

int main(int argc, const char** argv) {
    if(argc<3){
        printf("Please run the command with ./vr <streaming_socket_endpoint> <control_socket_endpoint>\n");
    }
    else if (argc>3){
        printf("Please run the command with ./vr <streaming_socket_endpoint> <control_socket_endpoint>\n");
    }
    else {
        STREAMING_SOCKET_ENDPOINT = argv[1];
        CONTROL_SOCKET_ENDPOINT = argv[2];
    }
    signal (SIGINT, quit_handler);
    v_initPre();

    zmq::context_t context(1);
    zmq::socket_t control_socket(context, ZMQ_PUB);

    control_socket.bind(CONTROL_SOCKET_ENDPOINT);
    control_socket.set(zmq::sockopt::conflate, 1);

    cout << "control socket bound" << endl;

#if STREAMING
    zmqpp::socket streaming_socket(context, zmqpp::socket_type::pull);
    streaming_socket.connect(STREAMING_SOCKET_ENDPOINT);
    // Don't wait for the simulation side to send the rendered image
    streaming_socket.set(zmqpp::socket_option::receive_timeout, 1);
    streaming_socket.set(zmqpp::socket_option::conflate, 1);

    cout << "streaming socket connected" << endl;
#endif

    int width = 1096 * 2;
    int height = 1176;
    size_t image_size = width * height * 3;
    char image[image_size];

    while (1) {

        draco::vr_teleop_msg m;
        transformVRPose();
    
        m.set_l_trigger(ctl[0].hold[vBUTTON_TRIGGER]);
        m.set_l_bump(ctl[0].hold[vBUTTON_SIDE]);
        m.set_l_button(ctl[0].hold[vBUTTON_MENU]);
        m.set_l_pad(ctl[0].hold[vBUTTON_PAD]);

        m.set_r_trigger(ctl[1].hold[vBUTTON_TRIGGER]);
        m.set_r_bump(ctl[1].hold[vBUTTON_SIDE]);
        m.set_r_button(ctl[1].hold[vBUTTON_MENU]);
        m.set_r_pad(ctl[1].hold[vBUTTON_PAD]);

        for (int i = 0; i < 3; ++i) {
            m.add_lh_pos(transformed_left_pos[i]);
            m.add_rh_pos(transformed_right_pos[i]);
        }

        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                m.add_lh_ori(transformed_left_ori(i, j));
                m.add_rh_ori(transformed_right_ori(i, j));
            }
        }

        string commands_str;
        try {
            m.SerializeToString(&commands_str);
        } catch (google::protobuf::FatalException e) {
            cout << e.message() << endl;
        }
        zmq::message_t zmq_msg(commands_str.size());
        memcpy ((void *) zmq_msg.data(), commands_str.c_str(),
                commands_str.size());
        control_socket.send(zmq_msg, zmq::send_flags::dontwait);

#if STREAMING
        streaming_socket.receive_raw(image, image_size);
        glActiveTexture(GL_TEXTURE2);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, image);

        v_render();
        v_update();
#endif
        v_update();
    }
}

