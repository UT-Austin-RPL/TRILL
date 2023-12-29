from collections import OrderedDict

import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.distributions as D
import torch.nn.functional as F
from robomimic.models.distributions import TanhWrappedDistribution
from robomimic.models.obs_nets import RNN_MIMO_MLP
from robomimic.models.policy_nets import RNNActorNetwork


class RNNHybridActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """

    def __init__(
        self,
        obs_shapes,
        trajectory_indices,
        discrete_classes,
        discrete_indices,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        self.discrete_classes = discrete_classes
        self.discrete_indices = discrete_indices
        self.trajectory_indices = trajectory_indices

        self.discrete_dim = len(discrete_indices)
        self.trajectory_dim = len(trajectory_indices)

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert (
            std_activation in self.activations
        ), "std_activation must be one of: {}; instead got: {}".format(
            self.activations.keys(), std_activation
        )
        self.std_activation = std_activation

        super(RNNHybridActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            # trajectory_dim=trajectory_dim,
            # discrete_dim=discrete_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.trajectory_dim),
            scale=(self.num_modes, self.trajectory_dim),
            logits=(self.num_modes,),
            discrete=(max(self.discrete_classes), self.discrete_dim),
        )

    def forward_train(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        outputs = RNN_MIMO_MLP.forward(
            self,
            obs=obs_dict,
            goal=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )

        if return_state:
            outputs, state = outputs
        else:
            state = None

        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]
        discrete = outputs["discrete"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(
            component_distribution, 1
        )  # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            dists = TanhWrappedDistribution(base_dist=dists, scale=1.0)

        discrete_logits = torch.softmax(discrete, dim=-2)

        if return_state:
            return (dists, discrete_logits), state
        else:
            return (dists, discrete_logits)

    def forward(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )
        if return_state:
            ad, state = out
            dists, discrete_logits = ad
            trajectory_sample = dists.sample()
            discrete_args = torch.argmax(discrete_logits, dim=-2)
            return (trajectory_sample, discrete_args), state
        else:
            dists, discrete_logits = out
            trajectory_sample = dists.sample()
            discrete_args = torch.argmax(discrete_logits, dim=-2)
            return (trajectory_sample, discrete_args)

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which
        is useful for computing quantities necessary at train-time, like
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )

        dists, discrete_logits = ad

        # to squeeze time dimension, make another action distribution
        assert dists.component_distribution.base_dist.loc.shape[1] == 1
        assert dists.component_distribution.base_dist.scale.shape[1] == 1
        assert dists.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=dists.component_distribution.base_dist.loc.squeeze(1),
            scale=dists.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(
            logits=dists.mixture_distribution.logits.squeeze(1)
        )
        dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return (dists, discrete_logits), state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )
        dist_act, discrete_act = acts
        assert dist_act.shape[1] == 1
        out = torch.zeros(
            (discrete_act.shape[0], self.discrete_dim + self.trajectory_dim),
            device=discrete_act.device,
        )
        # out[:, self.discrete_indices] = discrete_act[:, 0]
        out[:, self.discrete_indices] = discrete_act[:, 0].type(torch.float32)
        out[:, self.trajectory_indices] = dist_act[:, 0]
        return out, state

    def _to_string(self):
        """Info to pretty print."""
        msg = "trajectory_dim={}, discrete_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.trajectory_dim,
            self.discrete_dim,
            self.std_activation,
            self.low_noise_eval,
            self.num_modes,
            self.min_std,
        )
        return msg


class HierarchicalRNNDeterministicActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """

    def __init__(
        self,
        obs_shapes,
        trajectory_indices,
        discrete_trg_classes,
        discrete_trg_indices,
        discrete_con_classes,
        discrete_con_indices,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        self.use_tanh = use_tanh

        self.discrete_trg_classes = discrete_trg_classes
        self.discrete_trg_indices = discrete_trg_indices
        self.discrete_con_classes = discrete_con_classes
        self.discrete_con_indices = discrete_con_indices
        self.trajectory_indices = trajectory_indices

        self.discrete_con_dim = len(discrete_con_indices)
        self.discrete_trg_dim = len(discrete_trg_indices)
        self.trajectory_dim = len(trajectory_indices)

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }

        super(HierarchicalRNNDeterministicActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            # trajectory_dim=trajectory_dim,
            # discrete_dim=discrete_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            trajectory=(self.trajectory_dim,),
            trg_activation=(2, self.discrete_trg_dim),
            trg_discrete=(max(self.discrete_trg_classes) - 1, self.discrete_trg_dim),
            con_discrete=(max(self.discrete_con_classes), self.discrete_con_dim),
        )

    def forward_train(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        outputs = RNN_MIMO_MLP.forward(
            self,
            obs=obs_dict,
            goal=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )

        if return_state:
            outputs, state = outputs
        else:
            state = None

        trajectories = outputs["trajectory"]
        con_discrete = outputs["con_discrete"]
        trg_discrete = outputs["trg_discrete"]
        trg_activation = outputs["trg_activation"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            trajectories = torch.tanh(trajectories)

        trg_dists = D.Categorical(logits=trg_activation.transpose(-1, -2))
        trg_logits = torch.softmax(trg_discrete, dim=-2)
        con_logits = torch.softmax(con_discrete, dim=-2)

        if return_state:
            return (trajectories, con_logits, trg_dists, trg_logits), state
        else:
            return (trajectories, con_logits, trg_dists, trg_logits)

    def forward(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )
        if return_state:
            ad, state = out
            trajectories, continuous_logits, trigger_dists, trigger_logits = ad
            continuous_args = torch.argmax(continuous_logits, dim=-2)
            trigger_args = trigger_dists.sample() * (
                torch.argmax(trigger_logits, dim=-2) + 1
            )
            return (trajectories, continuous_args, trigger_args), state
        else:
            trajectories, continuous_logits, trigger_dists, trigger_logits = out
            continuous_args = torch.argmax(continuous_logits, dim=-2)
            trigger_args = trigger_dists.sample() * (
                torch.argmax(trigger_logits, dim=-2) + 1
            )
            return (trajectories, continuous_args, trigger_args)

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which
        is useful for computing quantities necessary at train-time, like
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )

        trajectories, con_logits, trg_dists, trg_logits = ad

        return (trajectories, con_logits, trg_dists, trg_logits), state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )
        traj_act, disc_con_act, disc_trg_act = acts
        assert traj_act.shape[1] == 1
        out = torch.zeros(
            (
                disc_con_act.shape[0],
                self.trajectory_dim + self.discrete_trg_dim + self.discrete_con_dim,
            ),
            device=disc_con_act.device,
        )
        # out[:, self.discrete_indices] = discrete_act[:, 0]
        out[:, self.discrete_con_indices] = disc_con_act[:, 0].type(torch.float32)
        out[:, self.discrete_trg_indices] = disc_trg_act[:, 0].type(torch.float32)
        out[:, self.trajectory_indices] = traj_act[:, 0]
        return out, state

    def _to_string(self):
        """Info to pretty print."""
        msg = "trajectory_dim={}, discrete_trg_dim={}, discrete_con_dim={}".format(
            self.trajectory_dim, self.discrete_trg_dim, self.discrete_con_dim
        )
        return msg


class RNNHumanoidActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """

    def __init__(
        self,
        obs_shapes,
        trajectory_indices,
        discrete_prob_classes,
        discrete_prob_indices,
        discrete_det_classes,
        discrete_det_indices,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        self.discrete_prob_classes = discrete_prob_classes
        self.discrete_prob_indices = discrete_prob_indices
        self.discrete_det_classes = discrete_det_classes
        self.discrete_det_indices = discrete_det_indices
        self.trajectory_indices = trajectory_indices

        self.discrete_det_dim = len(discrete_det_indices)
        self.discrete_prob_dim = len(discrete_prob_indices)
        self.trajectory_dim = len(trajectory_indices)

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert (
            std_activation in self.activations
        ), "std_activation must be one of: {}; instead got: {}".format(
            self.activations.keys(), std_activation
        )
        self.std_activation = std_activation

        super(RNNHumanoidActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            # trajectory_dim=trajectory_dim,
            # discrete_dim=discrete_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.trajectory_dim),
            scale=(self.num_modes, self.trajectory_dim),
            logits=(self.num_modes,),
            prob_discrete=(max(self.discrete_prob_classes), self.discrete_prob_dim),
            det_discrete=(max(self.discrete_det_classes), self.discrete_det_dim),
        )

    def forward_train(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        outputs = RNN_MIMO_MLP.forward(
            self,
            obs=obs_dict,
            goal=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )

        if return_state:
            outputs, state = outputs
        else:
            state = None

        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]
        det_discrete = outputs["det_discrete"]
        prob_discrete = outputs["prob_discrete"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(
            component_distribution, 1
        )  # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        traj_dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            traj_dists = TanhWrappedDistribution(base_dist=traj_dists, scale=1.0)

        disc_probs = torch.softmax(prob_discrete, dim=-2)
        disc_dists = D.Categorical(probs=disc_probs.transpose(-1, -2))
        disc_logits = torch.softmax(det_discrete, dim=-2)

        if return_state:
            return (traj_dists, disc_dists, disc_logits), state
        else:
            return (traj_dists, disc_dists, disc_logits)

    def forward(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )
        if return_state:
            ad, state = out
            trajectory_dists, discrete_dists, discrete_logits = ad
            trajectory_sample = trajectory_dists.sample()
            discrete_sample = discrete_dists.sample()
            discrete_args = torch.argmax(discrete_logits, dim=-2)
            return (trajectory_sample, discrete_sample, discrete_args), state
        else:
            trajectory_dists, discrete_dists, discrete_logits = out
            trajectory_sample = trajectory_dists.sample()
            discrete_sample = discrete_dists.sample()
            discrete_args = torch.argmax(discrete_logits, dim=-2)
            return (trajectory_sample, discrete_sample, discrete_args)

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which
        is useful for computing quantities necessary at train-time, like
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )

        trajectory_dists, discrete_dists, discrete_logits = ad

        # to squeeze time dimension, make another action distribution
        assert trajectory_dists.component_distribution.base_dist.loc.shape[1] == 1
        assert trajectory_dists.component_distribution.base_dist.scale.shape[1] == 1
        assert trajectory_dists.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=trajectory_dists.component_distribution.base_dist.loc.squeeze(1),
            scale=trajectory_dists.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(
            logits=trajectory_dists.mixture_distribution.logits.squeeze(1)
        )
        trajectory_dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return (trajectory_dists, discrete_dists, discrete_logits), state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )
        traj_act, disc_prob_act, disc_det_act = acts
        assert traj_act.shape[1] == 1
        out = torch.zeros(
            (
                disc_det_act.shape[0],
                self.trajectory_dim + self.discrete_prob_dim + self.discrete_det_dim,
            ),
            device=disc_det_act.device,
        )
        # out[:, self.discrete_indices] = discrete_act[:, 0]
        out[:, self.discrete_det_indices] = disc_det_act[:, 0].type(torch.float32)
        out[:, self.discrete_prob_indices] = disc_prob_act[:, 0].type(torch.float32)
        out[:, self.trajectory_indices] = traj_act[:, 0]
        return out, state

    def _to_string(self):
        """Info to pretty print."""
        msg = "trajectory_dim={}, discrete_prob_dim={}, discrete_det_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.trajectory_dim,
            self.discrete_prob_dim,
            self.discrete_det_dim,
            self.std_activation,
            self.low_noise_eval,
            self.num_modes,
            self.min_std,
        )
        return msg


class HierarchicalRNNHumanoidActorNetwork(RNNActorNetwork):
    """
    An RNN GMM policy network that predicts sequences of action distributions from observation sequences.
    """

    def __init__(
        self,
        obs_shapes,
        trajectory_indices,
        discrete_trg_classes,
        discrete_trg_indices,
        discrete_con_classes,
        discrete_con_indices,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        num_modes=5,
        min_std=0.01,
        std_activation="softplus",
        low_noise_eval=True,
        use_tanh=False,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            num_modes (int): number of GMM modes

            min_std (float): minimum std output from network

            std_activation (None or str): type of activation to use for std deviation. Options are:

                `'softplus'`: Softplus activation applied

                `'exp'`: Exp applied; this corresponds to network output being interpreted as log_std instead of std

            low_noise_eval (float): if True, model will sample from GMM with low std, so that
                one of the GMM modes will be sampled (approximately)

            use_tanh (bool): if True, use a tanh-Gaussian distribution

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """

        # parameters specific to GMM actor
        self.num_modes = num_modes
        self.min_std = min_std
        self.low_noise_eval = low_noise_eval
        self.use_tanh = use_tanh

        self.discrete_trg_classes = discrete_trg_classes
        self.discrete_trg_indices = discrete_trg_indices
        self.discrete_con_classes = discrete_con_classes
        self.discrete_con_indices = discrete_con_indices
        self.trajectory_indices = trajectory_indices

        self.discrete_con_dim = len(discrete_con_indices)
        self.discrete_trg_dim = len(discrete_trg_indices)
        self.trajectory_dim = len(trajectory_indices)

        # Define activations to use
        self.activations = {
            "softplus": F.softplus,
            "exp": torch.exp,
        }
        assert (
            std_activation in self.activations
        ), "std_activation must be one of: {}; instead got: {}".format(
            self.activations.keys(), std_activation
        )
        self.std_activation = std_activation

        super(HierarchicalRNNHumanoidActorNetwork, self).__init__(
            obs_shapes=obs_shapes,
            ac_dim=ac_dim,
            # trajectory_dim=trajectory_dim,
            # discrete_dim=discrete_dim,
            mlp_layer_dims=mlp_layer_dims,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            goal_shapes=goal_shapes,
            encoder_kwargs=encoder_kwargs,
        )

    def _get_output_shapes(self):
        """
        Tells @MIMO_MLP superclass about the output dictionary that should be generated
        at the last layer. Network outputs parameters of GMM distribution.
        """
        return OrderedDict(
            mean=(self.num_modes, self.trajectory_dim),
            scale=(self.num_modes, self.trajectory_dim),
            logits=(self.num_modes,),
            trg_activation=(2, self.discrete_trg_dim),
            trg_discrete=(max(self.discrete_trg_classes) - 1, self.discrete_trg_dim),
            con_discrete=(max(self.discrete_con_classes), self.discrete_con_dim),
        )

    def forward_train(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Return full GMM distribution, which is useful for computing
        quantities necessary at train-time, like log-likelihood, KL
        divergence, etc.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            dists (Distribution): sequence of GMM distributions over the timesteps
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(
                goal_dict, size=obs_dict[mod].shape[1], dim=1
            )

        outputs = RNN_MIMO_MLP.forward(
            self,
            obs=obs_dict,
            goal=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )

        if return_state:
            outputs, state = outputs
        else:
            state = None

        means = outputs["mean"]
        scales = outputs["scale"]
        logits = outputs["logits"]
        con_discrete = outputs["con_discrete"]
        trg_discrete = outputs["trg_discrete"]
        trg_activation = outputs["trg_activation"]

        # apply tanh squashing to mean if not using tanh-GMM to ensure means are in [-1, 1]
        if not self.use_tanh:
            means = torch.tanh(means)

        if self.low_noise_eval and (not self.training):
            # low-noise for all Gaussian dists
            scales = torch.ones_like(means) * 1e-4
        else:
            # post-process the scale accordingly
            scales = self.activations[self.std_activation](scales) + self.min_std

        # mixture components - make sure that `batch_shape` for the distribution is equal
        # to (batch_size, timesteps, num_modes) since MixtureSameFamily expects this shape
        component_distribution = D.Normal(loc=means, scale=scales)
        component_distribution = D.Independent(
            component_distribution, 1
        )  # shift action dim to event shape

        # unnormalized logits to categorical distribution for mixing the modes
        mixture_distribution = D.Categorical(logits=logits)

        traj_dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        if self.use_tanh:
            # Wrap distribution with Tanh
            traj_dists = TanhWrappedDistribution(base_dist=traj_dists, scale=1.0)

        trg_dists = D.Categorical(logits=trg_activation.transpose(-1, -2))
        trg_logits = torch.softmax(trg_discrete, dim=-2)
        con_logits = torch.softmax(con_discrete, dim=-2)

        if return_state:
            return (traj_dists, con_logits, trg_dists, trg_logits), state
        else:
            return (traj_dists, con_logits, trg_dists, trg_logits)

    def forward(
        self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False
    ):
        """
        Samples actions from the policy distribution.

        Args:
            obs_dict (dict): batch of observations
            goal_dict (dict): if not None, batch of goal observations

        Returns:
            action (torch.Tensor): batch of actions from policy distribution
        """
        out = self.forward_train(
            obs_dict=obs_dict,
            goal_dict=goal_dict,
            rnn_init_state=rnn_init_state,
            return_state=return_state,
        )
        if return_state:
            ad, state = out
            trajectory_dists, continuous_logits, trigger_dists, trigger_logits = ad
            trajectory_sample = trajectory_dists.sample()
            continuous_args = torch.argmax(continuous_logits, dim=-2)
            trigger_args = trigger_dists.sample() * (
                torch.argmax(trigger_logits, dim=-2) + 1
            )
            return (trajectory_sample, continuous_args, trigger_args), state
        else:
            trajectory_dists, continuous_logits, trigger_dists, trigger_logits = out
            trajectory_sample = trajectory_dists.sample()
            continuous_args = torch.argmax(continuous_logits, dim=-2)
            trigger_args = trigger_dists.sample() * (
                torch.argmax(trigger_logits, dim=-2) + 1
            )
            return (trajectory_sample, continuous_args, trigger_args)

    def forward_train_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get action GMM distribution, which
        is useful for computing quantities necessary at train-time, like
        log-likelihood, KL divergence, etc.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            ad (Distribution): GMM action distributions
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        ad, state = self.forward_train(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )

        traj_dists, con_logits, trg_dists, trg_logits = ad

        # to squeeze time dimension, make another action distribution
        assert trajectory_dists.component_distribution.base_dist.loc.shape[1] == 1
        assert trajectory_dists.component_distribution.base_dist.scale.shape[1] == 1
        assert trajectory_dists.mixture_distribution.logits.shape[1] == 1
        component_distribution = D.Normal(
            loc=trajectory_dists.component_distribution.base_dist.loc.squeeze(1),
            scale=trajectory_dists.component_distribution.base_dist.scale.squeeze(1),
        )
        component_distribution = D.Independent(component_distribution, 1)
        mixture_distribution = D.Categorical(
            logits=trajectory_dists.mixture_distribution.logits.squeeze(1)
        )
        trajectory_dists = D.MixtureSameFamily(
            mixture_distribution=mixture_distribution,
            component_distribution=component_distribution,
        )

        return (traj_dists, con_logits, trg_dists, trg_logits), state

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None):
        """
        Unroll RNN over single timestep to get sampled actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            acts (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        acts, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True
        )
        traj_act, disc_con_act, disc_trg_act = acts
        assert traj_act.shape[1] == 1
        out = torch.zeros(
            (
                disc_con_act.shape[0],
                self.trajectory_dim + self.discrete_trg_dim + self.discrete_con_dim,
            ),
            device=disc_con_act.device,
        )
        # out[:, self.discrete_indices] = discrete_act[:, 0]
        out[:, self.discrete_con_indices] = disc_con_act[:, 0].type(torch.float32)
        out[:, self.discrete_trg_indices] = disc_trg_act[:, 0].type(torch.float32)
        out[:, self.trajectory_indices] = traj_act[:, 0]
        return out, state

    def _to_string(self):
        """Info to pretty print."""
        msg = "trajectory_dim={}, discrete_trg_dim={}, discrete_con_dim={}, std_activation={}, low_noise_eval={}, num_nodes={}, min_std={}".format(
            self.trajectory_dim,
            self.discrete_trg_dim,
            self.discrete_con_dim,
            self.std_activation,
            self.low_noise_eval,
            self.num_modes,
            self.min_std,
        )
        return msg
