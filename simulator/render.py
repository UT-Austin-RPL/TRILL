# vr rendering
import mujoco
import numpy as np
import cv2
import time
from robosuite.utils.binding_utils import MjRenderContextOffscreen

RIGHTFORWARD_GRIPPER = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
RIGHTUP_GRIPPER = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

WIDTH = 1096 * 2
HEIGHT = 1176

class CV2Renderer():
    def __init__(self, device_id, sim, cam_name, width=480, height=360, depth=False, segmentation=False, save_path=None, gui=True) -> None:

        self._render_context = MjRenderContextOffscreen(sim, device_id=device_id)
        sim.add_render_context(self._render_context)

        self._width = width
        self._height = height
        self._depth = depth
        self._segmentation = segmentation
        self._cam_id = sim.model.camera_name2id(cam_name)
        self._gui = gui

        if save_path is not None:
            self._save_file = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))
        else:
            self._save_file = None

    def render(self):
        self._render_context.render(width=self._width, height=self._height, camera_id=self._cam_id, segmentation=self._segmentation)
        img = self._render_context.read_pixels(self._width, self._height, depth=self._depth, segmentation=self._segmentation)

        if self._save_file is not None:
            self._save_file.write(img[:,::-1,::-1])
        ###
        if self._gui:
            cv2.imshow('test', img[:,::-1,::-1])
            cv2.waitKey(1)
        return img

    def close(self):
        if self._save_file is not None:
            self._save_file.release()


class VRRenderer():

    def __init__(self, socket, sim, cam_name, width=WIDTH, height=HEIGHT, depth=False, segmentation=False) -> None:

        self._socket = socket

        self._sim = sim
        self._model = self._sim.model._model
        self._data = self._sim.data._data

        self._model.vis.global_.offwidth = width
        self._model.vis.global_.offheight = height

        self._cam = mujoco.MjvCamera()
        self._cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self._cam.fixedcamid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)

        self._glctx = mujoco.GLContext(max_width=WIDTH, max_height=HEIGHT)
        self._glctx.make_current()

        self._sim.forward()

        self._scn = mujoco.MjvScene(self._model, maxgeom=1000)
        self._vopt = mujoco.MjvOption()
        self._pert = mujoco.MjvPerturb()
        self._con = mujoco.MjrContext(self._model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self._scn.stereo = 2

        self._width = width
        self._height = height
        self._depth = depth
        self._segmentation = segmentation


    def render(self):
        mujoco.mjv_updateScene(self._model, self._data, self._vopt, self._pert, self._cam, mujoco.mjtCatBit.mjCAT_ALL, self._scn)
        # render in offscreen buffer
        #render_s = time.perf_counter()
        viewport = mujoco.MjrRect(0, 0, self._width, self._height)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self._con)
        mujoco.mjr_render(viewport=viewport, scn=self._scn, con=self._con)
        #read_s = time.perf_counter()

        #img = render_context_offscreen.read_pixels(width, height, depth=depth, segmentation=segmentation)
        rgb_img = np.empty((self._height, self._width, 3), dtype=np.uint8)
        mujoco.mjr_readPixels(rgb=rgb_img, depth=None, viewport=viewport, con=self._con)
        #read_e = time.perf_counter()
        self._socket.send(rgb_img.data)
        #cv2.imshow('test_python', rgb_img[:,:,::-1])
        #cv2.waitKey(1)
        #show_e = time.perf_counter()

        #print(f"rendering took {read_s - render_s:0.4f} seconds, reading pixels took {read_e - read_s:0.4f} seconds, while displaying it took {show_e-read_e:0.4f} seconds")

        return rgb_img

    def close(self):
        pass


def getVRPose(socket):
    FLOAT_SIZE = 4

    message = socket.recv()

    hmd_pos = np.frombuffer(message, dtype=np.float32, count=3, offset=0)
    hmd_mat = np.frombuffer(message, dtype=np.float32, count=9, offset=3 * FLOAT_SIZE).reshape((3, 3))
    left_pos = np.frombuffer(message, dtype=np.float32, count=3, offset=12 * FLOAT_SIZE)
    left_mat = np.frombuffer(message, dtype=np.float32, count=9, offset=15 * FLOAT_SIZE).reshape((3, 3))
    right_pos = np.frombuffer(message, dtype=np.float32, count=3, offset=24 * FLOAT_SIZE)
    right_mat = np.frombuffer(message, dtype=np.float32, count=9, offset=27 * FLOAT_SIZE).reshape((3, 3))
    
    # print(hmd_pos)
    # print(hmd_mat)
    # print(left_pos)
    # print(left_mat)
    # print(right_pos)
    # print(right_mat)

    local_left_pos = left_pos - hmd_pos
    local_right_pos = right_pos - hmd_pos
    mat_room2hmd = np.linalg.inv(hmd_mat)

    left_trigger = np.frombuffer(message, dtype=np.float32, count=1, offset=36 * FLOAT_SIZE)
    left_bump = np.frombuffer(message, dtype=np.float32, count=1, offset=37 * FLOAT_SIZE)
    left_button = np.frombuffer(message, dtype=np.float32, count=1, offset=38 * FLOAT_SIZE)
    left_pad = np.frombuffer(message, dtype=np.float32, count=1, offset=39 * FLOAT_SIZE)

    right_trigger = np.frombuffer(message, dtype=np.float32, count=1, offset=40 * FLOAT_SIZE)
    right_bump = np.frombuffer(message, dtype=np.float32, count=1, offset=41 * FLOAT_SIZE)
    right_button = np.frombuffer(message, dtype=np.float32, count=1, offset=42 * FLOAT_SIZE)
    right_pad = np.frombuffer(message, dtype=np.float32, count=1, offset=43 * FLOAT_SIZE)

    local_left_pos = mat_room2hmd @ local_left_pos 
    local_right_pos = mat_room2hmd @ local_right_pos

    transformed_left = -local_left_pos[[2, 0, 1]]
    transformed_left[2] = -transformed_left[2] + .2
    transformed_left[0] += .15
    transformed_right = -local_right_pos[[2, 0, 1]]
    transformed_right[0] += .15
    transformed_right[2] = -transformed_right[2] + .2

    left_orientation = mat_room2hmd @ left_mat
    right_orientation = mat_room2hmd @ right_mat

    # print(transformed_left)
    # print(transformed_right)
    # print(convert_orientation(left_orientation))
    # print(convert_orientation(right_orientation))

    return transformed_left, transformed_right, convert_orientation(left_orientation), convert_orientation(right_orientation), left_trigger, left_bump, left_button, left_pad, right_trigger, right_bump, right_button, right_pad


def convert_orientation(VR_orientation):
    T = np.array([[0, 0, 1], [1, 0, 0], [0, -1, 0]])
    B_inv = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    return T @ B_inv @ VR_orientation @ T.T


if __name__ == "__main__":
    pass
