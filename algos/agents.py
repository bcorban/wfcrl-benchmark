import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal, Beta
from wfcrl.extractors import FourierExtractor



def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent_normal(nn.Module): #Original agent implementation, unbounded action space
    def __init__(self, observation_space, action_space, hidden_layers, features_extractor_params={}):
        super().__init__()
        if features_extractor_params is None:
            features_extractor_params = {}
        action_dim = action_space.shape[0]

        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        features_extractor = FourierExtractor(observation_space, **features_extractor_params)

        input_layers = [features_extractor.features_dim] + list(hidden_layers)
        self.critic = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )
        self.actor = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], action_dim), std=1.0),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor(x)
        action_std = torch.ones_like(action_mean) * self.log_std.exp()
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.mode() if deterministic else distribution.rsample()
        return action, distribution.log_prob(action).sum(-1), distribution.entropy(), self.critic(x)



class Agent_bounded(nn.Module): #bound the action space by applying a tanh
    def __init__(self, observation_space, action_space, hidden_layers, features_extractor_params={}):
        super().__init__()
        action_dim = action_space.shape[0]
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

        self.log_std = nn.Parameter(torch.zeros(action_dim), requires_grad=True)
        features_extractor = FourierExtractor(observation_space, **features_extractor_params)

        input_layers = [features_extractor.features_dim] + list(hidden_layers)
        self.critic = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )
        self.actor = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], action_dim), std=1.0), nn.Tanh()
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        action_mean = self.actor(x)
        action_std = torch.ones_like(action_mean) * torch.clip(self.log_std,-1,1).exp()
        distribution = Normal(action_mean, action_std)
        if action is None:
            action = distribution.mode() if deterministic else distribution.rsample()

            # Squash the sampled action using Tanh to ensure it's within [-1, 1]
            squashed_action = torch.tanh(action)

            # Calculate log probability of the squashed action
            log_prob = distribution.log_prob(action).sum(-1)

            # Apply correction for the Tanh squashing (log_prob needs to account for Tanh change of variables)
            # This accounts for the change in the distribution when applying Tanh
            log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + 1e-6).sum(-1)
        else:
            squashed_action = (action-self.action_bias)/self.action_scale
            unsquashed_action = torch.atanh(torch.clamp(squashed_action, -0.999, 0.999))
            log_prob = distribution.log_prob(unsquashed_action).sum(-1)
            log_prob = log_prob - torch.log(1 - squashed_action.pow(2) + 1e-6).sum(-1)

        # return action, distribution.log_prob(action).sum(-1), distribution.entropy(), self.critic(x)
        return squashed_action * self.action_scale + self.action_bias, log_prob, distribution.entropy(), self.critic(x)

class Agent_beta(nn.Module):
    def __init__(self, observation_space, action_space, hidden_layers, features_extractor_params={}):
        super().__init__()
        action_dim = action_space.shape[0]
        self.alpha = 1
        self.beta = 1
        # Store action scale and bias for rescaling Beta samples to the desired action space
        self.register_buffer(
            "action_scale", torch.tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32)
        )

        features_extractor = FourierExtractor(observation_space, **features_extractor_params)

        input_layers = [features_extractor.features_dim] + list(hidden_layers)

        # Critic network (value function approximation)
        self.critic = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], 1), std=1.0),
        )

        # Actor network (outputs alpha and beta for the Beta distribution)
        self.actor = nn.Sequential(
            features_extractor,
            *[
                nn.Sequential(layer_init(nn.Linear(in_dim, out_dim)), nn.Tanh())
                for in_dim, out_dim in zip(input_layers[:-1], hidden_layers)
            ],
            layer_init(nn.Linear(input_layers[-1], 2 * action_dim), std=1.0),  # Output both alpha and beta
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, deterministic=False):
        # Get the raw output from the actor network
        alpha_beta = self.actor(x)

        # Split the output into alpha and beta parameters
        alpha, beta = torch.chunk(alpha_beta, 2, dim=-1)

        # Ensure alpha and beta are positive
        if not torch.isnan(alpha) : self.alpha = torch.nn.functional.softplus(alpha) + 1e-6  # Ensures alpha > 0
        else : self.alpha = self.alpha.detach()
        if not torch.isnan(beta) : self.beta = torch.nn.functional.softplus(beta) + 1e-6  # Ensures beta > 0
        else: self.beta = self.beta.detach()
        # Create a Beta distribution with the given alpha and beta
        # print(f'alpha : {alpha} : {self.alpha} - beta : {beta} : {self.beta}')

        distribution = Beta(self.alpha, self.beta)

        if action is None:
            # Sample action from Beta distribution
            action = distribution.mean if deterministic else distribution.rsample()

        else:  action = (((action - self.action_bias)/self.action_scale)+1)/2

        # Scale the action from [0, 1] to [low, high]
        scaled_action = self.action_scale * (2 * action - 1) + self.action_bias

        # Calculate log probability of the action
        log_prob = distribution.log_prob(action).sum(-1)

        # Return the scaled action, log probability, entropy, and value
        return scaled_action, log_prob, distribution.entropy(), self.critic(x)

AGENT_TYPE_DICT = {'beta' : Agent_beta, 'normal' : Agent_normal, 'bounded' :  Agent_bounded}