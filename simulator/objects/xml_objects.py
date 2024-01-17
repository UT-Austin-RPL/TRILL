import os

import numpy as np
from robosuite.models.objects import MujocoXMLObject
from robosuite.utils.mjcf_utils import (
    xml_path_completion,
    find_elements,
    array_to_string,
)

cwd = os.getcwd()
PATH_TO_OBJECT_MODELS = os.path.expanduser(cwd + "/models/objects")


class TableObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "table.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class ToasterObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "toaster.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class PotObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "pot.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class PotLidObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "pot_lid.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class DoorObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name, friction=None, damping=None, type="lock"):
        if type == "slide":
            xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "slide_door.xml")
        elif type == "hinge":
            xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "hinge_door.xml")
        elif type == "wheel":
            xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "wheel_door.xml")
        else:
            xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "locked_door.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )

        # Set relevant body names
        self.door_body = self.naming_prefix + "door"
        self.frame_body = self.naming_prefix + "frame"
        self.handle_body = self.naming_prefix + "handle"
        if type == "slide":
            self.door_joint = self.naming_prefix + "slide"
        else:
            self.door_joint = self.naming_prefix + "hinge"
        if type == "lock":
            self.lock_body = self.naming_prefix + "lock"
            self.latch_body = self.naming_prefix + "latch"

        self._locked = True
        self.friction = friction
        self.damping = damping
        if self.friction is not None:
            self._set_door_friction(self.friction)
        if self.damping is not None:
            self._set_door_damping(self.damping)

    def _set_door_friction(self, friction):
        """
        Helper function to override the door friction directly in the XML

        Args:
            friction (3-tuple of float): friction parameters to override the ones specified in the XML
        """
        hinge = find_elements(
            root=self.worldbody,
            tags="joint",
            attribs={"name": self.door_joint},
            return_first=True,
        )
        hinge.set("frictionloss", array_to_string(np.array([friction])))

    def _set_door_damping(self, damping):
        """
        Helper function to override the door friction directly in the XML

        Args:
            damping (float): damping parameter to override the ones specified in the XML
        """
        hinge = find_elements(
            root=self.worldbody,
            tags="joint",
            attribs={"name": self.door_joint},
            return_first=True,
        )
        hinge.set("damping", array_to_string(np.array([damping])))

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class BreadObject(MujocoXMLObject):
    """
    Bread loaf object (used in PickPlace)
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "bread.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class TrayObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "tray.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class HammerObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "hammer.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class LadleObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "ladle.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class ToolboxObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "toolbox.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class MicrowaveObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "microwave.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class StoveObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "stove.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class TargetObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "target.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class LaneObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "lane.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class LineObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "line.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class WallObject(MujocoXMLObject):
    """
    Wall (used in Wall)

    Args:
        name (str): Name of the wall
    """

    def __init__(self, name, color=None):
        if color is None:
            obj_path = "wall.xml"
        else:
            obj_path = "wall_{}.xml".format(color)

        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, obj_path)
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        return dic


class BrushObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "brush.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

        # Set relevant body names
        self.brush_body = self.naming_prefix + "body"
        self.furs_body = self.naming_prefix + "furs"

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class ParticleObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "particle.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )


class NailObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "nail.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=True,
        )

    @property
    def important_sites(self):
        """
        Returns:
            dict: In addition to any default sites for this object, also provides the following entries

                :`'handle'`: Name of door handle location site
        """
        # Get dict from super call and add to it
        dic = super().important_sites
        dic.update({"handle": self.naming_prefix + "handle"})
        return dic


class PegboardObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "pegboard.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[
                dict(
                    name="joint_x",
                    type="slide",
                    axis="1 0 0",
                    frictionloss="100",
                    damping="0.1",
                ),
                dict(
                    name="joint_y",
                    type="slide",
                    axis="0 1 0",
                    frictionloss="100",
                    damping="0.1",
                ),
                dict(
                    name="joint_yaw",
                    type="hinge",
                    axis="0 0 1",
                    frictionloss="100",
                    damping="0.1",
                ),
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )


class CartObject(MujocoXMLObject):
    """
    Door with handle (used in Door)

    Args:
        friction (3-tuple of float): friction parameters to override the ones specified in the XML
        damping (float): damping parameter to override the ones specified in the XML
        lock (bool): Whether to use the locked door variation object or not
    """

    def __init__(self, name):
        xml_path = os.path.join(PATH_TO_OBJECT_MODELS, "cart.xml")
        super().__init__(
            xml_path_completion(xml_path),
            name=name,
            joints=[
                dict(
                    name="joint_x",
                    type="slide",
                    axis="1 0 0",
                    frictionloss="0.5",
                    damping="320.",
                ),
                dict(
                    name="joint_y",
                    type="slide",
                    axis="0 1 0",
                    frictionloss="0.5",
                    damping="320.",
                ),
                dict(
                    name="joint_yaw",
                    type="hinge",
                    axis="0 0 1",
                    frictionloss="0.5",
                    damping="640.",
                ),
            ],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
