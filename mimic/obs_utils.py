import robomimic.utils.tensor_utils as TU
from robomimic.utils.obs_utils import Modality, process_frame, process_frame 
import robomimic.utils.tensor_utils as TensorUtils

class GrayModality(Modality):
    """
    Modality for RGB image observations
    """
    name = "gray"

    @classmethod
    def _default_obs_processor(cls, obs):
        """
        Given image fetched from dataset, process for network input. Converts array
        to float (from uint8), normalizes pixels from range [0, 255] to [0, 1], and channel swaps
        from (H, W, C) to (C, H, W).
        Args:
            obs (np.array or torch.Tensor): image array
        Returns:
            processed_obs (np.array or torch.Tensor): processed image
        """

        return process_frame(frame=obs, channel_dim=1, scale=255.)

        # out = process_frame(frame=obs, channel_dim=1, scale=255.)

        # original_stdout = sys.stdout # Save a reference to the original standard output
        # with open('/home/mingyo/gray.txt', 'a') as f:
        #    sys.stdout = f # Change the standard output to the file we created.
        #    print(out.shape, '\n', out[0])
        #    sys.stdout = original_stdout # Reset the standard output to its original value

        # return out

    @classmethod
    def _default_obs_unprocessor(cls, obs):
        """
        Given image prepared for network input, prepare for saving to dataset.
        Inverse of @process_frame.
        Args:
            obs (np.array or torch.Tensor): image array
        Returns:
            unprocessed_obs (np.array or torch.Tensor): image passed through
                inverse operation of @process_frame
        """
        return TU.to_uint8(unprocess_frame(frame=obs, channel_dim=1, scale=255.))

