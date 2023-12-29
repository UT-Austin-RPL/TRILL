from robosuite.models.arenas import EmptyArena

from .base import BaseEnv


class EmptyEnv(BaseEnv):
    def _load_model(self):
        # Create an environment
        super()._load_model()

        self.arena = EmptyArena()
        self.world.merge(self.arena)
