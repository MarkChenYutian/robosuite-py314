# Modifications Copyright 2022 The robosuite Authors
# Original Copyright 2018 The dm_control Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import atexit
import ctypes
import os

PYOPENGL_PLATFORM = os.environ.get("PYOPENGL_PLATFORM")

if not PYOPENGL_PLATFORM:
    os.environ["PYOPENGL_PLATFORM"] = "egl"
elif PYOPENGL_PLATFORM.lower() != "egl":
    raise ImportError(
        "Cannot use EGL rendering platform. "
        "The PYOPENGL_PLATFORM environment variable is set to {!r} "
        "(should be either unset or 'egl')."
    )

from mujoco.egl import egl_ext as EGL
from OpenGL import error


# From the EGL_EXT_device_query and EGL_NV_device_cuda extensions.
EGL_CUDA_DEVICE_NV = 0x323A
try:
    _eglQueryDeviceAttribEXT = ctypes.CFUNCTYPE(
        EGL.EGLBoolean, EGL.EGLDeviceEXT, ctypes.c_int, ctypes.POINTER(ctypes.c_ssize_t)
    )(EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
except TypeError:
    _eglQueryDeviceAttribEXT = None


def _cuda_ordinal(device):
    """CUDA ordinal of an EGL device, relative to CUDA_VISIBLE_DEVICES like torch's, or None if it has none."""
    if _eglQueryDeviceAttribEXT is None:
        return None
    value = ctypes.c_ssize_t(-1)
    if _eglQueryDeviceAttribEXT(device, EGL_CUDA_DEVICE_NV, ctypes.byref(value)) != EGL.EGL_TRUE:
        EGL.eglGetError()  # clear the error of a device without CUDA
        return None
    return value.value


def _requested_cuda_ordinal(device_id):
    """CUDA ordinal of device_id: a GPU id listed in CUDA_VISIBLE_DEVICES (any id if unset), -1 for the first visible."""
    if device_id == -1:
        return 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return device_id
    visible_ids = [x.strip() for x in visible.split(",")]
    if str(device_id) not in visible_ids:
        raise RuntimeError(f"Render device {device_id} is not in CUDA_VISIBLE_DEVICES={visible!r}.")
    return visible_ids.index(str(device_id))


def create_initialized_egl_device_display(device_id=0):
    """Creates an initialized EGL display directly on a device.

    The EGL device order generally differs from the CUDA order, so the device is the EGL device whose CUDA ordinal
    (EGL_NV_device_cuda) is that of device_id, see _requested_cuda_ordinal. MUJOCO_EGL_DEVICE_ID, if set, overrides
    it with a raw index into eglQueryDevicesEXT (as in mujoco.egl). Without CUDA devices, device_id (0 if -1) is
    used as a raw index.
    """
    all_devices = EGL.eglQueryDevicesEXT(max_devices=64)
    egl_device_id = os.environ.get("MUJOCO_EGL_DEVICE_ID", None)
    if egl_device_id is not None:
        if not (egl_device_id.isdigit() and int(egl_device_id) < len(all_devices)):
            raise RuntimeError(
                f"The MUJOCO_EGL_DEVICE_ID environment variable must be an integer "
                f"between 0 and {len(all_devices)-1} (inclusive), got {egl_device_id}."
            )
        candidates = all_devices[int(egl_device_id) : int(egl_device_id) + 1]
    else:
        ordinals = [_cuda_ordinal(device) for device in all_devices]
        if any(ordinal is not None for ordinal in ordinals):
            requested = _requested_cuda_ordinal(device_id)
            candidates = [device for device, ordinal in zip(all_devices, ordinals) if ordinal == requested]
            if not candidates:
                raise RuntimeError(f"No EGL device for CUDA device {requested}, the EGL devices have CUDA {ordinals}.")
        else:
            device_idx = max(device_id, 0)
            candidates = all_devices[device_idx : device_idx + 1]
    for device in candidates:
        display = EGL.eglGetPlatformDisplayEXT(EGL.EGL_PLATFORM_DEVICE_EXT, device, None)
        if display != EGL.EGL_NO_DISPLAY and EGL.eglGetError() == EGL.EGL_SUCCESS:
            # `eglInitialize` may or may not raise an exception on failure depending
            # on how PyOpenGL is configured. We therefore catch a `GLError` and also
            # manually check the output of `eglGetError()` here.
            try:
                initialized = EGL.eglInitialize(display, None, None)
            except error.GLError:
                pass
            else:
                if initialized == EGL.EGL_TRUE and EGL.eglGetError() == EGL.EGL_SUCCESS:
                    return display
    return EGL.EGL_NO_DISPLAY


global EGL_DISPLAY
EGL_DISPLAY = None

EGL_ATTRIBUTES = (
    EGL.EGL_RED_SIZE,
    8,
    EGL.EGL_GREEN_SIZE,
    8,
    EGL.EGL_BLUE_SIZE,
    8,
    EGL.EGL_ALPHA_SIZE,
    8,
    EGL.EGL_DEPTH_SIZE,
    24,
    EGL.EGL_STENCIL_SIZE,
    8,
    EGL.EGL_COLOR_BUFFER_TYPE,
    EGL.EGL_RGB_BUFFER,
    EGL.EGL_SURFACE_TYPE,
    EGL.EGL_PBUFFER_BIT,
    EGL.EGL_RENDERABLE_TYPE,
    EGL.EGL_OPENGL_BIT,
    EGL.EGL_NONE,
)


class EGLGLContext:
    """An EGL context for headless accelerated OpenGL rendering on GPU devices."""

    def __init__(self, max_width, max_height, device_id=0):

        del max_width, max_height  # unused
        num_configs = ctypes.c_long()
        config_size = 1
        config = EGL.EGLConfig()
        EGL.eglReleaseThread()
        global EGL_DISPLAY
        if EGL_DISPLAY is None:
            # only initialize for the first time
            EGL_DISPLAY = create_initialized_egl_device_display(device_id=device_id)
            if EGL_DISPLAY == EGL.EGL_NO_DISPLAY:
                raise ImportError(
                    "Cannot initialize a EGL device display. This likely means that your EGL "
                    "driver does not support the PLATFORM_DEVICE extension, which is "
                    "required for creating a headless rendering context."
                )
            atexit.register(EGL.eglTerminate, EGL_DISPLAY)
        EGL.eglChooseConfig(EGL_DISPLAY, EGL_ATTRIBUTES, ctypes.byref(config), config_size, num_configs)
        if num_configs.value < 1:
            raise RuntimeError(
                "EGL failed to find a framebuffer configuration that matches the "
                "desired attributes: {}".format(EGL_ATTRIBUTES)
            )
        EGL.eglBindAPI(EGL.EGL_OPENGL_API)
        self._context = EGL.eglCreateContext(EGL_DISPLAY, config, EGL.EGL_NO_CONTEXT, None)
        if not self._context:
            raise RuntimeError("Cannot create an EGL context.")

    def make_current(self):
        if not EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, self._context):
            raise RuntimeError("Failed to make the EGL context current.")

    def free(self):
        """Frees resources associated with this context."""
        if self._context:
            current_context = EGL.eglGetCurrentContext()
            if current_context and self._context.address == current_context.address:
                EGL.eglMakeCurrent(EGL_DISPLAY, EGL.EGL_NO_SURFACE, EGL.EGL_NO_SURFACE, EGL.EGL_NO_CONTEXT)
            EGL.eglDestroyContext(EGL_DISPLAY, self._context)
            EGL.eglReleaseThread()
        self._context = None

    def __del__(self):
        self.free()
