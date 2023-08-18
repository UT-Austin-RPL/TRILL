//-----------------------------------//
//  This file is part of MuJoCo.     //
//  Copyright (C) 2016 Roboti LLC.   //
//-----------------------------------//

#include "mujoco.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"

#include "GL/glew.h"
#include "glfw3.h"

#include <openvr.h>
using namespace vr;


//-------------------------------- MuJoCo global data -----------------------------------

// MuJoCo model and data
mjModel* m = 0;
mjData* d = 0;

// MuJoCo visualization
mjvScene scn;
mjvOption vopt;
mjvPerturb pert;
mjrContext con;
GLFWwindow* window;
double frametime = 0;


//-------------------------------- MuJoCo functions -------------------------------------

// load model, init simulation and rendering; return 0 if error, 1 if ok
int initMuJoCo(const char* filename, int width2, int height)
{
    // init GLFW
    if( !glfwInit() )
    {
        printf("Could not initialize GLFW\n");
        return 0;
    }
    glfwWindowHint(GLFW_SAMPLES, 0);
    glfwWindowHint(GLFW_DOUBLEBUFFER, 1);
    glfwWindowHint(GLFW_RESIZABLE, 0);
    window = glfwCreateWindow(width2/4, height/2, "MuJoCo VR", NULL, NULL);
    if( !window )
    {
        printf("Could not create GLFW window\n");
        return 0;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);
    if( glewInit()!=GLEW_OK )
        return 0;

    // activate
    if( !mj_activate("mjkey.txt") )
        return 0;

    // load and compile
    char error[1000] = "Could not load binary model";
    if( strlen(filename)>4 && !strcmp(filename+strlen(filename)-4, ".mjb") )
        m = mj_loadModel(filename, 0, 0);
    else
        m = mj_loadXML(filename, 0, error, 1000);
    if( !m )
    {
        printf("%s\n", error);
        return 0;
    }

    // make data, run one computation to initialize all fields
    d = mj_makeData(m);
    mj_forward(m, d);

    // set offscreen buffer size to match HMD
    m->vis.global.offwidth = width2;
    m->vis.global.offheight = height;
    m->vis.quality.offsamples = 8;

    // initialize MuJoCo visualization
    mjv_makeScene(&scn, 1000);
    mjv_defaultOption(&vopt);
    mjv_defaultPerturb(&pert);
    mjr_defaultContext(&con);
    mjr_makeContext(m, &con, 100);

    // initialize model transform
    scn.enabletransform = 1;
    scn.translate[1] = -0.5;
    scn.translate[2] = -0.5;
    scn.rotate[0] = (float)cos(-0.25*mjPI);
    scn.rotate[1] = (float)sin(-0.25*mjPI);
    scn.scale = 1;

    // stereo mode
    scn.stereo = mjSTEREO_SIDEBYSIDE;

    return 1;
}


// deallocate everything and deactivate
void closeMuJoCo(void)
{
    mj_deleteData(d);
    mj_deleteModel(m);
    mjr_freeContext(&con);
    mjv_freeScene(&scn);
    mj_deactivate();
}


// keyboard
void keyboard(GLFWwindow* window, int key, int scancode, int act, int mods)
{
    // require model
    if( !m )
        return;

    // do not act on release
    if( act==GLFW_RELEASE )
        return;

    switch( key )
    {
    case ';':                           // previous frame mode
        vopt.frame = mjMAX(0, vopt.frame-1);
        break;

    case '\'':                          // next frame mode
        vopt.frame = mjMIN(mjNFRAME-1, vopt.frame+1);
        break;

    case '.':                           // previous label mode
        vopt.label = mjMAX(0, vopt.label-1);
        break;

    case '/':                           // next label mode
        vopt.label = mjMIN(mjNLABEL-1, vopt.label+1);
        break;

    case GLFW_KEY_BACKSPACE:
        mj_resetData(m, d);
        mj_forward(m, d);
        frametime = 0;
        break;

    default:
        // toggle visualization flag
        for( int i=0; i<mjNVISFLAG; i++ )
            if( key==mjVISSTRING[i][2][0] )
                vopt.flags[i] = !vopt.flags[i];

        // toggle rendering flag
        for( int i=0; i<mjNRNDFLAG; i++ )
            if( key==mjRNDSTRING[i][2][0] )
                scn.flags[i] = !scn.flags[i];

        // toggle geom/site group
        for( int i=0; i<mjNGROUP; i++ )
            if( key==i+'0')
            {
                if( mods & GLFW_MOD_SHIFT )
                    vopt.sitegroup[i] = !vopt.sitegroup[i];
                else
                    vopt.geomgroup[i] = !vopt.geomgroup[i];
            }
    }
}



//-------------------------------- vr definitions and global data -----------------------

// buttons
enum
{
    vBUTTON_TRIGGER = 0,
    vBUTTON_SIDE,
    vBUTTON_MENU,
    vBUTTON_PAD,

    vNBUTTON
};


// tools
enum
{
    vTOOL_MOVE = 0,
    vTOOL_PULL,

    vNTOOL
};


// tool names
const char* toolName[vNTOOL] = {
    "move and scale world",
    "pull selected body"
};


// all data related to one controller
struct _vController_t
{
    // constant properties
    int id;                         // controller device id; -1: not used
    int idtrigger;                  // trigger axis id
    int idpad;                      // trackpad axis id
    float rgba[4];                  // controller color

    // modes
    bool valid;                     // controller is connected and visible
    bool touch[vNBUTTON];           // button is touched
    bool hold[vNBUTTON];            // button is held
    int tool;                       // active tool (vTOOL_XXX)
    int body;                       // selected MuJoCo body

    // pose in room (raw data)
    float roompos[3];               // position in room
    float roommat[9];               // orientation matrix in room

    // pose in model (transformed)
    mjtNum pos[3];                  // position
    mjtNum quat[4];                 // orientation

    // target pose
    mjtNum targetpos[3];            // target position
    mjtNum targetquat[4];           // target orientation

    // offset for remote tools
    mjtNum relpos[3];               // position offset
    mjtNum relquat[4];              // orientation offset

    // analog axis input
    float triggerpos;               // trigger position
    float padpos[2];                // pad 2D position

    // old data, used to compute delta
    float oldroompos[3];            // old room position
    float oldroommat[9];            // old room orientation
    float oldtriggerpos;            // old trigger position
    float oldpadpos[2];             // old pad 2D position

    // text message
    char message[100];              // message
    double messagestart;            // time when display started
    double messageduration;         // duration for display
};
typedef struct _vController_t vController_t;


// all data related to HMD
struct _vHMD_t
{
    // constant properties
    IVRSystem *system;              // opaque pointer returned by VR_Init
    uint32_t width, height;         // recommended image size per eye
    int id;                         // hmd device id
    unsigned int idtex;             // OpenGL texture id for Submit
    float eyeoffset[2][3];          // head-to-eye offsets (assume no rotation)

    // pose in room (raw data)
    float roompos[3];               // position
    float roommat[9];               // orientation matrix
};
typedef struct _vHMD_t vHMD_t;


// vr global variables
vHMD_t hmd;
vController_t ctl[2];


//-------------------------------- vr functions -----------------------------------------

// init vr: before MuJoCo init
void v_initPre(void)
{
    int n, i;

    // initialize runtime
    EVRInitError err = VRInitError_None;
    hmd.system = VR_Init(&err, VRApplication_Scene);
    if ( err!=VRInitError_None )
        mju_error_s("Could not init VR runtime: %s", VR_GetVRInitErrorAsEnglishDescription(err));

    // initialize compositor, set to Standing
    if( !VRCompositor() )
    {
        VR_Shutdown();
        mju_error("Could not init Compositor");
    }
    VRCompositor()->SetTrackingSpace(TrackingUniverseStanding);

    // get recommended image size
    hmd.system->GetRecommendedRenderTargetSize(&hmd.width, &hmd.height);

    // check all devices, find hmd and controllers
    int cnt = 0;
    hmd.id = -1;
    ctl[0].id = -1;
    ctl[1].id = -1;
    for( n=0; n<k_unMaxTrackedDeviceCount; n++ )
    {
        ETrackedDeviceClass cls = hmd.system->GetTrackedDeviceClass(n);

        // found HMD
        if( cls==TrackedDeviceClass_HMD )
            hmd.id = n;

        // found Controller: max 2 supported
        else if( cls==TrackedDeviceClass_Controller && cnt<2 )
        {
            ctl[cnt].id = n;
            cnt++;
        }
    }

    // require HMD and at least one controller
    if( hmd.id<0 || ctl[0].id<0 )
        mju_error("Expected HMD and at least one Controller");

    // init HMD pose data
    for( n=0; n<9; n++ )
    {
        hmd.roommat[n] = 0;
        if( n<3 )
            hmd.roompos[n] = 0;
    }
    hmd.roommat[0] = 1;
    hmd.roommat[4] = 1;
    hmd.roommat[8] = 1;

    // get HMD eye-to-head offsets (no rotation)
    for( n=0; n<2; n++ )
    {
        HmdMatrix34_t tmp = hmd.system->GetEyeToHeadTransform((EVREye)n);
        hmd.eyeoffset[n][0] = tmp.m[0][3];
        hmd.eyeoffset[n][1] = tmp.m[1][3];
        hmd.eyeoffset[n][2] = tmp.m[2][3];
    }

    // init controller data
    for( n=0; n<2; n++ )
        if( ctl[n].id>=0 )
        {
            // get axis ids
            ctl[n].idtrigger = -1;
            ctl[n].idpad = -1;
            for( i=0; i<k_unControllerStateAxisCount; i++ )
            {
                // get property
                int prop = hmd.system->GetInt32TrackedDeviceProperty(ctl[n].id, 
                    (ETrackedDeviceProperty)(Prop_Axis0Type_Int32 + i));

                // assign id if matching
                if( prop==k_eControllerAxis_Trigger )
                    ctl[n].idtrigger = i;
                else if( prop==k_eControllerAxis_TrackPad )
                    ctl[n].idpad = i;
            }

            // make sure all ids were found
            if( ctl[n].idtrigger<0 || ctl[n].idpad<0 )
                mju_error("Trigger or Pad axis not found");

            // set colors
            if( n==0 )
            {
                ctl[n].rgba[0] = 0.8f;
                ctl[n].rgba[1] = 0.2f;
                ctl[n].rgba[2] = 0.2f;
                ctl[n].rgba[3] = 0.6f;
            }
            else
            {
                ctl[n].rgba[0] = 0.2f;
                ctl[n].rgba[1] = 0.8f;
                ctl[n].rgba[2] = 0.2f;
                ctl[n].rgba[3] = 0.6f;
            }

            // clear state
            ctl[n].valid = false;
            ctl[n].tool = (n==0 ? vTOOL_MOVE : vTOOL_PULL);
            ctl[n].body = 0;
            ctl[n].message[0] = 0;
            for( i=0; i<vNBUTTON; i++ )
            {
                ctl[n].touch[i] = false;
                ctl[n].hold[i] = false;
            }
        }
}


// init vr: after MuJoCo init
void v_initPost(void)
{
    // set MuJoCo OpenGL frustum to match Vive
    for( int n=0; n<2; n++ )
    {
        // get frustum from vr
        float left, right, top, bottom, znear = 0.05f, zfar = 50.0f;
        hmd.system->GetProjectionRaw((EVREye)n, &left, &right, &top, &bottom);

        // set in MuJoCo
        scn.camera[n].frustum_bottom = -bottom*znear;
        scn.camera[n].frustum_top = -top*znear;
        scn.camera[n].frustum_center = 0.5f*(left + right)*znear;
        scn.camera[n].frustum_near = znear;
        scn.camera[n].frustum_far = zfar;
    }

    // create vr texture
    glActiveTexture(GL_TEXTURE2);
    glGenTextures(1, &hmd.idtex);
    glBindTexture(GL_TEXTURE_2D, hmd.idtex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 2*hmd.width, hmd.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
}


// copy one pose from vr to our format
void v_copyPose(const TrackedDevicePose_t* pose, float* roompos, float* roommat)
{
    // nothing to do if not tracked
    if( !pose->bPoseIsValid )
        return;

    // pointer to data for convenience
    const HmdMatrix34_t* p = &pose->mDeviceToAbsoluteTracking;

    // raw data: room
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


// make default abstract geom
void v_defaultGeom(mjvGeom* geom)
{
    geom->type = mjGEOM_NONE;
    geom->dataid = -1;
    geom->objtype = mjOBJ_UNKNOWN;
    geom->objid = -1;
    geom->category = mjCAT_DECOR;
    geom->texid = -1;
    geom->texuniform = 0;
    geom->texrepeat[0] = 1;
    geom->texrepeat[1] = 1;
    geom->emission = 0;
    geom->specular = 0.5;
    geom->shininess = 0.5;
    geom->reflectance = 0;
    geom->label[0] = 0;
}


// update vr poses and controller states
void v_update(void)
{
    int n, i;
    mjvGeom* g;

    // get new poses
    TrackedDevicePose_t poses[k_unMaxTrackedDeviceCount];
    VRCompositor()->WaitGetPoses(poses, k_unMaxTrackedDeviceCount, NULL, 0 );

    // copy hmd pose
    v_copyPose(poses+hmd.id, hmd.roompos, hmd.roommat);

    // adjust OpenGL scene cameras to match hmd pose
    for( n=0; n<2; n++ )
    {
        // assign position, apply eye-to-head offset
        for( i=0; i<3; i++ )
            scn.camera[n].pos[i] = hmd.roompos[i] +
                hmd.eyeoffset[n][0]*hmd.roommat[3*i+0] + 
                hmd.eyeoffset[n][1]*hmd.roommat[3*i+1] + 
                hmd.eyeoffset[n][2]*hmd.roommat[3*i+2];

        // assign forward and up
        scn.camera[n].forward[0] = -hmd.roommat[2];
        scn.camera[n].forward[1] = -hmd.roommat[5];
        scn.camera[n].forward[2] = -hmd.roommat[8];
        scn.camera[n].up[0] = hmd.roommat[1];
        scn.camera[n].up[1] = hmd.roommat[4];
        scn.camera[n].up[2] = hmd.roommat[7];
    }

    // update controllers
    for( n=0; n<2; n++ )
        if( ctl[n].id>=0 )
        {
            // copy pose, set valid flag
            v_copyPose(poses+ctl[n].id, ctl[n].roompos, ctl[n].roommat);
            ctl[n].valid = poses[ctl[n].id].bPoseIsValid && poses[ctl[n].id].bDeviceIsConnected;

            // transform from room to model pose
            if( ctl[n].valid )
            {
                mjtNum rpos[3], rmat[9], rquat[4];
                mju_f2n(rpos, ctl[n].roompos, 3);
                mju_f2n(rmat, ctl[n].roommat, 9);
                mju_mat2Quat(rquat, rmat);
                mjv_room2model(ctl[n].pos, ctl[n].quat, rpos, rquat, &scn);
            }

            // update axis data
            VRControllerState_t state;
            hmd.system->GetControllerState(ctl[n].id, &state);
            ctl[n].triggerpos = state.rAxis[ctl[n].idtrigger].x;
            ctl[n].padpos[0] = state.rAxis[ctl[n].idpad].x;
            ctl[n].padpos[1] = state.rAxis[ctl[n].idpad].y;
        }

    // process events: button and touch only
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
            int button = vBUTTON_TRIGGER;
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

            // process event
            switch( evt.eventType )
            {
            case VREvent_ButtonPress:
                ctl[n].hold[button] = true;

                // trigger button: save relative pose
                if( button==vBUTTON_TRIGGER )
                {
                    // reset old pose
                    for( i=0; i<9; i++ )
                    {
                        ctl[n].oldroommat[i] = ctl[n].roommat[i];
                        if( i<3 )
                            ctl[n].oldroompos[i] = ctl[n].roompos[i];
                    }

                    // record relative pose
                    mjtNum negp[3], negq[4], xiquat[4];
                    mju_mulQuat(xiquat, d->xquat+4*ctl[n].body, m->body_iquat+4*ctl[n].body);
                    mju_negPose(negp, negq, ctl[n].pos, ctl[n].quat);
                    mju_mulPose(ctl[n].relpos, ctl[n].relquat, negp, negq, d->xipos+3*ctl[n].body, xiquat);
                }

                // menu button: change tool and show message
                else if( button==vBUTTON_MENU )
                {
                    ctl[n].tool = (ctl[n].tool + 1) % vNTOOL;
                    ctl[n].messageduration = 1;
                    ctl[n].messagestart = glfwGetTime();
                    strcpy(ctl[n].message, toolName[ctl[n].tool]);
                }

                // pad button: change selection and show message
                else if( button==vBUTTON_PAD && ctl[n].tool!=vTOOL_MOVE )
                {
                    if( ctl[n].padpos[1]>0 )
                        ctl[n].body = mjMAX(0, ctl[n].body-1);
                    else
                        ctl[n].body = mjMIN(m->nbody-1, ctl[n].body+1);

                    ctl[n].messageduration = 1;
                    ctl[n].messagestart = glfwGetTime();
                    const char* name = mj_id2name(m, mjOBJ_BODY, ctl[n].body);
                    if( name )
                        sprintf(ctl[n].message, "body '%s'", name);
                    else
                        sprintf(ctl[n].message, "body %d", ctl[n].body);
                }

                // side button: reserved for user
                else if( button==vBUTTON_SIDE )
                {
                    // user can trigger custom action here
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

    // finish controller update, after processing events
    for( n=0; n<2; n++ )
        if( ctl[n].id>=0 )
        {
            // update target pose
            if( ctl[n].hold[vBUTTON_TRIGGER] && ctl[n].tool!=vTOOL_MOVE )
                mju_mulPose(ctl[n].targetpos, ctl[n].targetquat, 
                    ctl[n].pos, ctl[n].quat, ctl[n].relpos, ctl[n].relquat);
            else
            {
                mju_copy3(ctl[n].targetpos, ctl[n].pos);
                mju_copy(ctl[n].targetquat, ctl[n].quat, 4);
            }

            // render controller
            if( scn.ngeom<scn.maxgeom )
            {
                float sclrgb = ctl[n].hold[vBUTTON_TRIGGER] ? 1 : 0.5f;
                g = scn.geoms + scn.ngeom;
                v_defaultGeom(g);
                g->size[0] = 0.03f / scn.scale;
                g->size[1] = 0.02f / scn.scale;
                g->size[2] = 0.04f / scn.scale;
                g->rgba[0] = ctl[n].rgba[0] * sclrgb;
                g->rgba[1] = ctl[n].rgba[1] * sclrgb;
                g->rgba[2] = ctl[n].rgba[2] * sclrgb;
                g->rgba[3] = ctl[n].rgba[3];
                mju_n2f(g->pos, ctl[n].targetpos, 3);
                mjtNum mat[9];
                mju_quat2Mat(mat, ctl[n].targetquat);
                mju_n2f(g->mat, mat, 9);

                if( ctl[n].tool==vTOOL_MOVE )
                {
                    g->type = mjGEOM_ARROW2;
                    g->size[0] = g->size[1] = 0.01f / scn.scale;
                    g->size[2] = 0.08f / scn.scale;
                }
                else
                    g->type = mjGEOM_BOX;

                if( ctl[n].message[0] && glfwGetTime()-ctl[n].messagestart < ctl[n].messageduration)
                    strcpy(g->label, ctl[n].message);

                scn.ngeom++;
            }

            // render connector for pull
            if( scn.ngeom<scn.maxgeom && ctl[n].tool==vTOOL_PULL && 
                ctl[n].body>0 && ctl[n].hold[vBUTTON_TRIGGER] )
            {
                mjtNum* p1 = ctl[n].targetpos;
                mjtNum* p2 = d->xipos + 3*ctl[n].body;
                mjtNum dif[3], mid[3], quat[4], mat[9];
                mju_add3(mid, p1, p2);
                mju_scl3(mid, mid, 0.5);
                mju_sub3(dif, p2, p1);

                g = scn.geoms + scn.ngeom;
                v_defaultGeom(g);
                g->type = mjGEOM_CAPSULE;
                g->size[0] = g->size[1] = (float)(0.5 * m->vis.scale.constraint * m->stat.meansize);
                g->size[2] = (float)(0.5 * mju_dist3(p1, p2));
                g->rgba[0] = ctl[n].rgba[0];
                g->rgba[1] = ctl[n].rgba[1];
                g->rgba[2] = ctl[n].rgba[2];
                g->rgba[3] = 1;

                mju_n2f(g->pos, mid, 3);
                mju_quatZ2Vec(quat, dif);
                mju_quat2Mat(mat, quat);
                mju_n2f(g->mat, mat, 9);

                scn.ngeom++;
            }

            // color selected body
            if( ctl[n].body>0 )
            {
                // search all geoms
                for( i=0; i<scn.ngeom; i++ )            
                {
                    g = scn.geoms + i;

                    // detect geoms belonging to selected body
                    if( g->category==mjCAT_DYNAMIC && g->objtype==mjOBJ_GEOM && 
                        m->geom_bodyid[g->objid]==ctl[n].body )
                    {
                        // common selection
                        if( ctl[0].valid && ctl[1].valid && ctl[0].body==ctl[1].body )
                        {
                            g->rgba[0] = (ctl[0].rgba[0] + ctl[1].rgba[0]);
                            g->rgba[1] = (ctl[0].rgba[1] + ctl[1].rgba[1]);
                            g->rgba[2] = (ctl[0].rgba[2] + ctl[1].rgba[2]);
                        }

                        // separate selections
                        else
                        {
                            g->rgba[0] = ctl[n].rgba[0];
                            g->rgba[1] = ctl[n].rgba[1];
                            g->rgba[2] = ctl[n].rgba[2];
                        }
                    }
                }
            }
        }

    // apply move and scale (other tools applied before mj_step)
    for( n=0; n<2; n++ )
        if( ctl[n].id>=0 && ctl[n].valid && 
            ctl[n].tool==vTOOL_MOVE && ctl[n].hold[vBUTTON_TRIGGER] )
        {
            // apply scaling and reset
            if( ctl[n].touch[vBUTTON_PAD] )
            {
                scn.scale += logf(1 + scn.scale/3) * (ctl[n].padpos[1] - ctl[n].oldpadpos[1]);
                if( scn.scale<0.01f )
                    scn.scale = 0.01f;
                else if( scn.scale>100.0f )
                    scn.scale = 100.0f;

                ctl[n].oldpadpos[1] = ctl[n].padpos[1];
            }

            // apply translation and reset
            for( i=0; i<3; i++ )
            {
                scn.translate[i] += ctl[n].roompos[i] - ctl[n].oldroompos[i];
                ctl[n].oldroompos[i] = ctl[n].roompos[i];
            }

            // compute rotation quaternion around vertical axis (room y)
            mjtNum mat[9], oldmat[9], difmat[9], difquat[4], vel[3], yaxis[3]={0,1,0};
            mju_f2n(mat, ctl[n].roommat, 9);
            mju_f2n(oldmat, ctl[n].oldroommat, 9);
            mju_mulMatMatT(difmat, mat, oldmat, 3, 3, 3);
            mju_mat2Quat(difquat, difmat);
            mju_quat2Vel(vel, difquat, 1);
            mju_axisAngle2Quat(difquat, yaxis, vel[1]);

            // apply rotation
            mjtNum qold[4], qnew[4];
            mju_f2n(qold, scn.rotate, 4);
            mju_mulQuat(qnew, difquat, qold);
            mju_normalize(qnew, 4);
            mju_n2f(scn.rotate, qnew, 4);

            // adjust translation so as to center rotation at controller
            float dx = scn.translate[0] - ctl[n].roompos[0];
            float dz = scn.translate[2] - ctl[n].roompos[2];
            float ca = (float)mju_cos(vel[1]);
            float sa = (float)mju_sin(vel[1]);
            scn.translate[0] = ctl[n].roompos[0] + dx*ca + dz*sa;
            scn.translate[2] = ctl[n].roompos[2] - dx*sa + dz*ca;
            
            // reset rotation
            for( i=0; i<9; i++ )
                ctl[n].oldroommat[i] = ctl[n].roommat[i];
        }
}


// render to vr and window
void v_render(void)
{
    // resolve multi-sample offscreen buffer
    glBindFramebuffer(GL_READ_FRAMEBUFFER, con.offFBO);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, con.offFBO_r);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);
    glBlitFramebuffer(0, 0, 2*hmd.width, hmd.height,
                      0, 0, 2*hmd.width, hmd.height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    // blit to window, left only, window is half-size
    glBindFramebuffer(GL_READ_FRAMEBUFFER, con.offFBO_r);
    glReadBuffer(GL_COLOR_ATTACHMENT0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glDrawBuffer(con.windowDoublebuffer ? GL_BACK : GL_FRONT);
    glBlitFramebuffer(0, 0, hmd.width, hmd.height,
                      0, 0, hmd.width/2, hmd.height/2,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);

    // blit to vr texture
    glActiveTexture(GL_TEXTURE2);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, con.offFBO_r);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, hmd.idtex, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT1);
    glBlitFramebuffer(0, 0, 2*hmd.width, hmd.height,
                      0, 0, 2*hmd.width, hmd.height,
                      GL_COLOR_BUFFER_BIT, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, 0, 0);
    glDrawBuffer(GL_COLOR_ATTACHMENT0);

    // submit to vr
    const VRTextureBounds_t boundLeft = {0, 0, 0.5, 1};
    const VRTextureBounds_t boundRight = {0.5, 0, 1, 1};
    Texture_t vTex = {(void*)hmd.idtex, API_OpenGL, ColorSpace_Gamma};
    VRCompositor()->Submit(Eye_Left, &vTex, &boundLeft);
    VRCompositor()->Submit(Eye_Right, &vTex, &boundRight);

    // swap if window is double-buffered, flush just in case
    if( con.windowDoublebuffer )
        glfwSwapBuffers(window);
    glFlush();
}


// close vr
void v_close(void)
{
    glDeleteTextures(1, &hmd.idtex);

    // crash inside OpenVR ???
    // VR_Shutdown();
}



//-------------------------------- main function ----------------------------------------

int main(int argc, const char** argv)
{
    char filename[100];

    // get filename from command line or iteractively
    if( argc==2 )
        strcpy(filename, argv[1]);
    else
    {
        printf("Enter MuJoCo model file: ");
        scanf("%s", filename);
    }

    // pre-initialize vr
    v_initPre();

    // initialize MuJoCo, with image size from vr
    if( !initMuJoCo(filename, (int)(2*hmd.width), (int)hmd.height) )
        return 0;

    // post-initialize vr
    v_initPost();

    // set keyboard callback
    glfwSetKeyCallback(window, keyboard);

    // main loop
    double lasttm = glfwGetTime(), FPS = 90;
    frametime = d->time;
    while( !glfwWindowShouldClose(window) )
    {
        // render new frame if it is time, or if simulation was reset
        if( (d->time-frametime)>1.0/FPS || d->time<frametime )
        {
            // create abstract scene
            mjv_updateScene(m, d, &vopt, NULL, NULL, mjCAT_ALL, &scn);

            // update vr poses and controller states
            v_update();

            // render in offscreen buffer
            mjrRect viewFull = {0, 0, 2*(int)hmd.width, (int)hmd.height};
            mjr_setBuffer(mjFB_OFFSCREEN, &con);
            mjr_render(viewFull, &scn, &con);

            // show FPS (window only, hmd clips it)
            FPS = 0.9*FPS + 0.1/(glfwGetTime() - lasttm);
            lasttm = glfwGetTime();
            char fpsinfo[20];
            sprintf(fpsinfo, "FPS %.0f", FPS);
            mjr_overlay(mjFONT_BIG, mjGRID_BOTTOMLEFT, viewFull, fpsinfo, NULL, &con);

            // render to vr and window
            v_render();

            // save simulation time
            frametime = d->time;  
        }

        // apply controller perturbations
        mju_zero(d->xfrc_applied, 6*m->nbody);
        for( int n=0; n<2; n++ )
            if( ctl[n].valid && ctl[n].tool==vTOOL_PULL && 
                ctl[n].body>0 && ctl[n].hold[vBUTTON_TRIGGER] )
            {
                // perpare mjvPerturb object
                pert.active = mjPERT_TRANSLATE | mjPERT_ROTATE;
                pert.select = ctl[n].body;
                mju_copy3(pert.refpos, ctl[n].targetpos);
                mju_copy(pert.refquat, ctl[n].targetquat, 4);

                // apply
                mjv_applyPerturbPose(m, d, &pert, 0);
                mjv_applyPerturbForce(m, d, &pert);
            }

        // simulate
        mj_step(m, d);

        // update GUI
        glfwPollEvents();
    }

    // close
    v_close();
    closeMuJoCo();
    glfwTerminate();

    return 1;
}
