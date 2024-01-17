from collections import OrderedDict

import torch
import numpy as np
from robomimic.algo import RolloutPolicy
from robomimic.algo.bc import *
from robomimic.utils.file_utils import (
    maybe_dict_from_checkpoint,
    config_from_checkpoint,
    algo_name_from_checkpoint,
)  # noqa: E402

from .policy_nets import *


def algo_factory(algo_name, config, obs_key_shapes, ac_dim, device):
    """
    Factory function for creating algorithms based on the algorithm name and config.

    Args:
        algo_name (str): the algorithm name

        config (BaseConfig instance): config object

        obs_key_shapes (OrderedDict): dictionary that maps observation keys to shapes

        ac_dim (int): dimension of action space

        device (torch.Device): where the algo should live (i.e. cpu, gpu)
    """

    # @algo_name is included as an arg to be explicit, but make sure it matches the config
    assert algo_name == config.algo_name and algo_name == "bc"

    algo_config = config.algo
    obs_config = config.observation

    gaussian_enabled = "gaussian" in algo_config and algo_config.gaussian.enabled
    gmm_enabled = "gmm" in algo_config and algo_config.gmm.enabled
    vae_enabled = "vae" in algo_config and algo_config.vae.enabled

    if algo_config.rnn.enabled:
        if gmm_enabled:
            if len(algo_config.gmm.discrete_indices) == 0:
                assert len(algo_config.gmm.discrete_indices) == len(
                    algo_config.gmm.discrete_classes
                )
                algo_cls = BC_RNN_GMM
            elif sum(algo_config.gmm.discrete_trg_config) != 0:
                algo_cls = BC_HIERARCHICAL_RNN_HUMANOID
            else:
                algo_cls = BC_RNN_HYBRID
        else:
            if len(algo_config.gmm.discrete_indices) == 0:
                assert len(algo_config.gmm.discrete_indices) == len(
                    algo_config.gmm.discrete_classes
                )
                algo_cls = BC_RNN
            elif sum(algo_config.gmm.discrete_trg_config) != 0:
                algo_cls = BC_HIERARCHICAL_RNN_DETERMINISTIC
            else:
                raise NotImplementedError
    else:
        assert sum([gaussian_enabled, gmm_enabled, vae_enabled]) <= 1
        if gaussian_enabled:
            algo_cls = BC_Gaussian
        if gmm_enabled:
            algo_cls = BC_GMM
        if vae_enabled:
            algo_cls = BC_VAE
        else:
            algo_cls = BC

    # create algo instance
    return algo_cls(
        algo_config=algo_config,
        obs_config=obs_config,
        global_config=config,
        obs_key_shapes=obs_key_shapes,
        ac_dim=ac_dim,
        device=device,
    )


def policy_from_checkpoint(device=None, ckpt_path=None, ckpt_dict=None, verbose=False):
    """
    This function restores a trained policy from a checkpoint file or
    loaded model dictionary.

    Args:
        device (torch.device): if provided, put model on this device

        ckpt_path (str): Path to checkpoint file. Only needed if not providing @ckpt_dict.

        ckpt_dict(dict): Loaded model checkpoint dictionary. Only needed if not providing @ckpt_path.

        verbose (bool): if True, include print statements

    Returns:
        model (RolloutPolicy): instance of Algo that has the saved weights from
            the checkpoint file, and also acts as a policy that can easily
            interact with an environment in a training loop

        ckpt_dict (dict): loaded checkpoint dictionary (convenient to avoid
            re-loading checkpoint from disk multiple times)
    """
    ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path, ckpt_dict=ckpt_dict)

    # algo name and config from model dict
    algo_name, _ = algo_name_from_checkpoint(ckpt_dict=ckpt_dict)
    config, _ = config_from_checkpoint(
        algo_name=algo_name, ckpt_dict=ckpt_dict, verbose=verbose
    )

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # shape meta from model dict to get info needed to create model
    shape_meta = ckpt_dict["shape_metadata"]

    # maybe restore observation normalization stats
    obs_normalization_stats = ckpt_dict.get("obs_normalization_stats", None)
    if obs_normalization_stats is not None:
        assert config.train.hdf5_normalize_obs
        for m in obs_normalization_stats:
            for k in obs_normalization_stats[m]:
                obs_normalization_stats[m][k] = np.array(obs_normalization_stats[m][k])

    if device is None:
        # get torch device
        device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # create model and load weights
    model = algo_factory(
        algo_name,
        config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    model.deserialize(ckpt_dict["model"])
    model.set_eval()
    model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)
    if verbose:
        print("============= Loaded Policy =============")
        print(model)
    return model, ckpt_dict


class BC_RNN_HYBRID(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled
        assert len(self.algo_config.gmm.discrete_indices) != 0

        self.discrete_indices = self.algo_config.gmm.discrete_indices
        self.discrete_classes = self.algo_config.gmm.discrete_classes

        self.trajectory_indices = [
            idx for idx in range(self.ac_dim) if idx not in self.discrete_indices
        ]

        print("Discrete class: ", self.discrete_classes)
        print("Discrete indices: ", self.discrete_indices)
        print("Trajectory indices: ", self.trajectory_indices)

        self.discrete_dim = len(self.discrete_indices)
        self.trajctory_dim = len(self.trajectory_indices)
        self.ac_dim = self.discrete_dim + self.trajctory_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = RNNHybridActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            trajectory_indices=self.trajectory_indices,
            discrete_classes=self.discrete_classes,
            discrete_indices=self.discrete_indices,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        pred_dists, pred_discrete = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        demo_trajectory = batch["actions"][..., self.trajectory_indices]
        demo_discrete = torch.round(batch["actions"][..., self.discrete_indices]).type(
            torch.long
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(pred_dists.batch_shape) == 2  # [B, T]
        log_probs = pred_dists.log_prob(demo_trajectory)
        cross_entropy = nn.CrossEntropyLoss()(
            pred_discrete.flatten(end_dim=1), demo_discrete.flatten(end_dim=1)
        )

        # original_stdout = sys.stdout # Save a reference to the original standard output
        # with open('/home/mingyo/test.txt', 'a') as f:
        #    sys.stdout = f # Change the standard output to the file we created.
        #    print(pred_discrete.flatten(end_dim=1)[0], demo_discrete.flatten(end_dim=1)[0])
        #    sys.stdout = original_stdout # Reset the standard output to its original value

        predictions = OrderedDict(
            log_probs=log_probs,
            cross_entropy=cross_entropy,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        log_probs = predictions["log_probs"].mean()
        cross_entropy = predictions["cross_entropy"]
        action_loss = (
            -self.algo_config.loss.log_probs_weight * log_probs
            + self.algo_config.loss.cross_entropy_weight * cross_entropy
        )
        return OrderedDict(
            cross_etropy=cross_entropy,
            log_probs=log_probs,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Log_Likelihood"] = info["losses"]["log_probs"].item()
        log["Cross_Entropy"] = info["losses"]["cross_etropy"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_RNN_HUMANOID(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled
        assert len(self.algo_config.gmm.discrete_indices) != 0
        assert sum(self.algo_config.gmm.discrete_prob_config) != 0
        assert len(self.algo_config.gmm.discrete_indices) == len(
            self.algo_config.gmm.discrete_prob_config
        )

        self.discrete_det_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if not self.algo_config.gmm.discrete_prob_config[idx]
        ]
        self.discrete_prob_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if self.algo_config.gmm.discrete_prob_config[idx]
        ]
        self.discrete_det_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if not self.algo_config.gmm.discrete_prob_config[idx]
        ]
        self.discrete_prob_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if self.algo_config.gmm.discrete_prob_config[idx]
        ]
        self.trajectory_indices = [
            idx
            for idx in range(self.ac_dim)
            if idx not in self.discrete_det_indices
            and idx not in self.discrete_prob_indices
        ]
        print("Discrete deterministic class: ", self.discrete_det_classes)
        print("Discrete deterministic indices: ", self.discrete_det_indices)
        print("Discrete probabilistic class: ", self.discrete_prob_classes)
        print("Discrete probabilistic indices: ", self.discrete_prob_indices)
        print("Trajectory indices: ", self.trajectory_indices)

        self.discrete_det_dim = len(self.discrete_det_indices)
        self.discrete_prob_dim = len(self.discrete_prob_indices)
        self.trajctory_dim = len(self.trajectory_indices)
        self.ac_dim = (
            self.discrete_det_dim + self.discrete_prob_dim + self.trajctory_dim
        )

        self.nets = nn.ModuleDict()
        self.nets["policy"] = RNNHumanoidActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            trajectory_indices=self.trajectory_indices,
            discrete_det_classes=self.discrete_det_classes,
            discrete_det_indices=self.discrete_det_indices,
            discrete_prob_classes=self.discrete_prob_classes,
            discrete_prob_indices=self.discrete_prob_indices,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        pred_traj_dists, pred_disc_dists, pred_disc_logits = self.nets[
            "policy"
        ].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        demo_trajectory = batch["actions"][..., self.trajectory_indices]
        demo_det_discrete = torch.round(
            batch["actions"][..., self.discrete_det_indices]
        ).type(torch.long)
        demo_prob_discrete = torch.round(
            batch["actions"][..., self.discrete_prob_indices]
        ).type(torch.long)

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(pred_traj_dists.batch_shape) == 2  # [B, T]
        traj_log_probs = pred_traj_dists.log_prob(demo_trajectory)
        disc_log_probs = pred_disc_dists.log_prob(demo_prob_discrete)
        disc_cross_entropy = nn.CrossEntropyLoss()(
            pred_disc_logits.flatten(end_dim=1), demo_det_discrete.flatten(end_dim=1)
        )

        # original_stdout = sys.stdout # Save a reference to the original standard output
        # with open('/home/mingyo/normalize.txt', 'a') as f:
        #     sys.stdout = f # Change the standard output to the file we created.
        #     print(pred_disc_dists.probs.flatten(end_dim=1)[0], demo_det_discrete.flatten(end_dim=1)[0])
        #     sys.stdout = original_stdout # Reset the standard output to its original value

        predictions = OrderedDict(
            disc_cross_entropy=disc_cross_entropy,
            disc_log_probs=disc_log_probs,
            traj_log_probs=traj_log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        traj_log_probs = predictions["traj_log_probs"].mean()
        disc_log_probs = predictions["disc_log_probs"].mean()
        disc_cross_entropy = predictions["disc_cross_entropy"]
        action_loss = (
            -self.algo_config.loss.log_probs_weight * traj_log_probs
            - self.algo_config.loss.log_probs_weight * 100.0 * disc_log_probs
            + self.algo_config.loss.cross_entropy_weight * disc_cross_entropy
        )
        return OrderedDict(
            disc_cross_entropy=disc_cross_entropy,
            disc_log_probs=disc_log_probs,
            traj_log_probs=traj_log_probs,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Trajectory_Log_Likelihood"] = info["losses"]["traj_log_probs"].item()
        log["Discrete_Log_Likelihood"] = info["losses"]["disc_log_probs"].item()
        log["Discrete_Cross_Entropy"] = info["losses"]["disc_cross_entropy"].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_HIERARCHICAL_RNN_DETERMINISTIC(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.rnn.enabled
        assert len(self.algo_config.gmm.discrete_indices) != 0
        assert sum(self.algo_config.gmm.discrete_trg_config) != 0
        assert len(self.algo_config.gmm.discrete_indices) == len(
            self.algo_config.gmm.discrete_trg_config
        )

        self.discrete_con_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if not self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_trg_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_con_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if not self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_trg_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.trajectory_indices = [
            idx
            for idx in range(self.ac_dim)
            if idx not in self.discrete_con_indices
            and idx not in self.discrete_trg_indices
        ]
        print("Discrete contiuous action class: ", self.discrete_con_classes)
        print("Discrete contiuous action indices: ", self.discrete_con_indices)
        print("Discrete trigger action class: ", self.discrete_trg_classes)
        print("Discrete trigger action indices: ", self.discrete_trg_indices)
        print("Trajectory indices: ", self.trajectory_indices)

        self.discrete_con_dim = len(self.discrete_con_indices)
        self.discrete_trg_dim = len(self.discrete_trg_indices)
        self.trajctory_dim = len(self.trajectory_indices)
        self.ac_dim = self.discrete_con_dim + self.discrete_trg_dim + self.trajctory_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = HierarchicalRNNDeterministicActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            trajectory_indices=self.trajectory_indices,
            discrete_con_classes=self.discrete_con_classes,
            discrete_con_indices=self.discrete_con_indices,
            discrete_trg_classes=self.discrete_trg_classes,
            discrete_trg_indices=self.discrete_trg_indices,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        pred_trajectory, pred_con_logits, pred_trg_dists, pred_trg_logits = self.nets[
            "policy"
        ].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        pred_con_logits_flatten = pred_con_logits.flatten(end_dim=1)
        pred_trg_logits_flatten = pred_trg_logits.flatten(end_dim=1)

        demo_trajectory = batch["actions"][..., self.trajectory_indices]
        demo_con_discrete_flatten = (
            torch.round(batch["actions"][..., self.discrete_con_indices])
            .type(torch.long)
            .flatten(end_dim=1)
        )

        demo_trg_discrete = torch.round(
            batch["actions"][..., self.discrete_trg_indices]
        ).type(torch.long)
        demo_trg_activation = torch.where(demo_trg_discrete == 0, 0, 1).type(torch.long)
        demo_trg_discrete_flatten = demo_trg_discrete.flatten(end_dim=1)
        demo_trg_value_indices = demo_trg_discrete_flatten.nonzero(as_tuple=True)[
            0
        ].tolist()

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        # assert len(pred_trajectory.batch_shape) == 2 # [B, T]
        traj_l2_loss = nn.MSELoss()(pred_trajectory, demo_trajectory)
        con_cross_entropy = nn.CrossEntropyLoss()(
            pred_con_logits_flatten, demo_con_discrete_flatten
        )
        activ_log_probs = pred_trg_dists.log_prob(demo_trg_activation)

        if len(demo_trg_value_indices) != 0:
            value_cross_entropy = nn.CrossEntropyLoss()(
                pred_trg_logits_flatten[demo_trg_value_indices],
                demo_trg_discrete_flatten[demo_trg_value_indices] - 1,
            )
        else:
            value_cross_entropy = torch.tensor([0.0], device=self.device)

        predictions = OrderedDict(
            con_cross_entropy=con_cross_entropy,
            activ_log_probs=activ_log_probs,
            value_cross_entropy=value_cross_entropy,
            traj_l2_loss=traj_l2_loss,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        traj_l2_loss = predictions["traj_l2_loss"].mean()
        con_cross_entropy = predictions["con_cross_entropy"]
        activ_log_probs = 200.0 * predictions["activ_log_probs"].mean()
        value_cross_entropy = predictions["value_cross_entropy"]
        action_loss = (
            +self.algo_config.loss.l2_weight * traj_l2_loss
            + self.algo_config.loss.cross_entropy_weight * con_cross_entropy
            - self.algo_config.loss.log_probs_weight * activ_log_probs
            + self.algo_config.loss.cross_entropy_weight * value_cross_entropy
        )
        return OrderedDict(
            con_cross_entropy=con_cross_entropy,
            activ_log_probs=activ_log_probs,
            value_cross_entropy=value_cross_entropy,
            traj_l2_loss=traj_l2_loss,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Trajectory_L2_Loss"] = info["losses"]["traj_l2_loss"].item()
        log["Trigger_Log_Likelihood"] = info["losses"]["activ_log_probs"].item()
        log["Trigger_Value_Cross_Entropy"] = info["losses"][
            "value_cross_entropy"
        ].item()
        log["Continuous_Discrete_Cross_Entropy"] = info["losses"][
            "con_cross_entropy"
        ].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log


class BC_HIERARCHICAL_RNN_HUMANOID(BC_RNN):
    """
    BC training with an RNN GMM policy.
    """

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.gmm.enabled
        assert self.algo_config.rnn.enabled
        assert len(self.algo_config.gmm.discrete_indices) != 0
        assert sum(self.algo_config.gmm.discrete_trg_config) != 0
        assert len(self.algo_config.gmm.discrete_indices) == len(
            self.algo_config.gmm.discrete_trg_config
        )

        self.discrete_con_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if not self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_trg_classes = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_classes)
            if self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_con_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if not self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.discrete_trg_indices = [
            act_idx
            for idx, act_idx in enumerate(self.algo_config.gmm.discrete_indices)
            if self.algo_config.gmm.discrete_trg_config[idx]
        ]
        self.trajectory_indices = [
            idx
            for idx in range(self.ac_dim)
            if idx not in self.discrete_con_indices
            and idx not in self.discrete_trg_indices
        ]
        print("Discrete contiuous action class: ", self.discrete_con_classes)
        print("Discrete contiuous action indices: ", self.discrete_con_indices)
        print("Discrete trigger action class: ", self.discrete_trg_classes)
        print("Discrete trigger action indices: ", self.discrete_trg_indices)
        print("Trajectory indices: ", self.trajectory_indices)

        self.discrete_con_dim = len(self.discrete_con_indices)
        self.discrete_trg_dim = len(self.discrete_trg_indices)
        self.trajctory_dim = len(self.trajectory_indices)
        self.ac_dim = self.discrete_con_dim + self.discrete_trg_dim + self.trajctory_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = HierarchicalRNNHumanoidActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            trajectory_indices=self.trajectory_indices,
            discrete_con_classes=self.discrete_con_classes,
            discrete_con_indices=self.discrete_con_indices,
            discrete_trg_classes=self.discrete_trg_classes,
            discrete_trg_indices=self.discrete_trg_indices,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(
                self.obs_config.encoder
            ),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

        self._rnn_hidden_state = None
        self._rnn_horizon = self.algo_config.rnn.horizon
        self._rnn_counter = 0
        self._rnn_is_open_loop = self.algo_config.rnn.get("open_loop", False)

        self.nets = self.nets.float().to(self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """
        pred_traj_dists, pred_con_logits, pred_trg_dists, pred_trg_logits = self.nets[
            "policy"
        ].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"],
        )

        pred_con_logits_flatten = pred_con_logits.flatten(end_dim=1)
        pred_trg_logits_flatten = pred_trg_logits.flatten(end_dim=1)

        demo_trajectory = batch["actions"][..., self.trajectory_indices]
        demo_con_discrete_flatten = (
            torch.round(batch["actions"][..., self.discrete_con_indices])
            .type(torch.long)
            .flatten(end_dim=1)
        )

        demo_trg_discrete = torch.round(
            batch["actions"][..., self.discrete_trg_indices]
        ).type(torch.long)
        demo_trg_activation = torch.where(demo_trg_discrete == 0, 0, 1).type(torch.long)
        demo_trg_discrete_flatten = demo_trg_discrete.flatten(end_dim=1)
        demo_trg_value_indices = demo_trg_discrete_flatten.nonzero(as_tuple=True)[
            0
        ].tolist()

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(pred_traj_dists.batch_shape) == 2  # [B, T]
        traj_log_probs = pred_traj_dists.log_prob(demo_trajectory)
        con_cross_entropy = nn.CrossEntropyLoss()(
            pred_con_logits_flatten, demo_con_discrete_flatten
        )
        activ_log_probs = pred_trg_dists.log_prob(demo_trg_activation)

        if len(demo_trg_value_indices) != 0:
            value_cross_entropy = nn.CrossEntropyLoss()(
                pred_trg_logits_flatten[demo_trg_value_indices],
                demo_trg_discrete_flatten[demo_trg_value_indices] - 1,
            )
        else:
            value_cross_entropy = torch.tensor([0.0], device=self.device)

        predictions = OrderedDict(
            con_cross_entropy=con_cross_entropy,
            activ_log_probs=activ_log_probs,
            value_cross_entropy=value_cross_entropy,
            traj_log_probs=traj_log_probs,
        )
        return predictions

    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """

        # loss is just negative log-likelihood of action targets
        traj_log_probs = predictions["traj_log_probs"].mean()
        con_cross_entropy = predictions["con_cross_entropy"]
        activ_log_probs = 200.0 * predictions["activ_log_probs"].mean()
        value_cross_entropy = predictions["value_cross_entropy"]
        action_loss = (
            -self.algo_config.loss.log_probs_weight * traj_log_probs
            + self.algo_config.loss.cross_entropy_weight * con_cross_entropy
            - self.algo_config.loss.log_probs_weight * activ_log_probs
            + self.algo_config.loss.cross_entropy_weight * value_cross_entropy
        )
        return OrderedDict(
            con_cross_entropy=con_cross_entropy,
            activ_log_probs=activ_log_probs,
            value_cross_entropy=value_cross_entropy,
            traj_log_probs=traj_log_probs,
            action_loss=action_loss,
        )

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = PolicyAlgo.log_info(self, info)
        log["Loss"] = info["losses"]["action_loss"].item()
        log["Trajectory_Log_Likelihood"] = info["losses"]["traj_log_probs"].item()
        log["Trigger_Log_Likelihood"] = info["losses"]["activ_log_probs"].item()
        log["Trigger_Value_Cross_Entropy"] = info["losses"][
            "value_cross_entropy"
        ].item()
        log["Continuous_Discrete_Cross_Entropy"] = info["losses"][
            "con_cross_entropy"
        ].item()
        if "policy_grad_norms" in info:
            log["Policy_Grad_Norms"] = info["policy_grad_norms"]
        return log
