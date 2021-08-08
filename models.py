from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gym
import torch as th

class EmbeddinsFeatureExtractor(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, EMBEDDING_DIM=2, features_dim: int = 256, ):
        super(EmbeddinsFeatureExtractor, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        self.base_embeding = th.nn.Embedding(4, EMBEDDING_DIM)

        self.linear_1 = th.nn.Sequential(th.nn.Linear(EMBEDDING_DIM * observation_space.shape[0], features_dim), th.nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        
        return self.linear_1(self.base_embeding(observations.int()).flatten(start_dim=1))