import os
import sys

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion

cwd = os.getcwd()
sys.path.append(cwd)

PATH_TO_ARENA_XML = os.path.expanduser(cwd + "/models/arenas/navigation_arena.xml")


class NavigationArena(Arena):
    """Empty workspace."""

    def __init__(self):
        super().__init__(xml_path_completion(PATH_TO_ARENA_XML))
