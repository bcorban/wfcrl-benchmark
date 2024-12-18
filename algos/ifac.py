import os
import sys
import random
import time
from dataclasses import dataclass
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions import Normal

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'wfcrl-env')))

from wfcrl.rewards import FilteredStep, TrackReward
from wfcrl import environments as envs
from wfcrl.rewards import *
from wfcrl.extractors import *
from utils import LocalSummaryWriter, plot_env_history
from agents import *

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "Floris_xp"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    #environment, objective and control settings
    env_id: str = "Dec_Turb3_Row1_Floris" #""Turb32_Row5_Floris
    """the id of the environment"""
    total_timesteps: int = 2000
    """total timesteps of the experiments"""
    control: str = 'ct'
    """type of control used ('yaw', 'pitch', 'ct')"""
    task: str = 'track'
    """objective ('max' or 'track')"""
    reward : str = 'shaped'
    """reward function type for maximisation task"""
    p_ref =2.5
    """reference power for simple tracking signal creation"""
    action_bound: float = 0.1
    """Bounds on the action space"""
    action_max: int = 0.8
    """maximum control angle value in state space"""
    action_min: int = 0.2
    """minimum control angle value in state space"""
    agent_type : str = 'normal'
    "Agent type (normal, bounded or beta)"

    #Hyperparameters
    learning_rate: float = 7e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.75
    """the discount factor gamma"""
    norm_adv: bool = False
    """Toggles advantages normalization"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    num_envs: int = 1
    """the number of parallel game environments"""

    # DFAC arguments
    pretrained_models: str = None
    """Path to pretrained models"""
    reward_tol: float = 0.00005
    """Tolerance threshold for reward function"""
    kl_coef:  float = 0.0
    """Weighing coefficient for KL term in loss """ 
    policy: str = "base"
    """Type of policy"""
    adaptive_explo: bool = False
    """Use counter_based exploration"""
    multi_scale: bool = False
    """Use multi scale algorithm"""
    hidden_layer_nn: Union[bool, tuple[int]] = False #(81,)
    """number of neurons in hidden layer"""
    yaw_max: int = 40
    """maximum absolute yawing in state space""" 
    fourier_order: int = 8
    """order of Fourier basis"""
    fourier_maxdim: int = 81
    """maximum dimension of Fourier basis"""
    fourier_hyper:  bool = False
    """use hypernet arch to learn Fourier basis"""
    fourier_learnable: bool = False
    """learnable Fourier basis"""
    fourier_hyper_arch: tuple = (81,)
    """Architecture of Fourier Hyper"""
    debug: bool = False
    """debug mode saves monitoring logs during training"""

    # to be filled in runtime
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    num_agents: int = 1
    """the number of agents in the environment"""
    reward_shaping: str = ""
    """Toggle learning rate annealing for policy and value networks"""
    
def make_action(action):
    return {args.control: action.cpu().numpy()}


if __name__ == "__main__":
    args = tyro.cli(Args)

    controls = {
        args.control: (args.action_min, args.action_max, args.action_bound)
    }

    track_power = [args.p_ref + 0.2 * (i>60000) for i in
        range(args.total_timesteps+1)]

    reward_shaper = FilteredStep(threshold=args.reward_tol) if args.task == 'max' else TrackReward(
            threshold=args.reward_tol,
            reference=track_power)

    env = envs.make(
        args.env_id,
        controls=controls, 
        max_num_steps=args.total_timesteps, 
        reward_shaper=reward_shaper
    )

    args.num_steps = args.total_timesteps
    args.num_agents = env.num_turbines
    args.reward_shaping = reward_shaper.name
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        # os.environ["HTTPS_PROXY"] = "http://irsrvpxw1-std:8082"
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = LocalSummaryWriter(f"runs/{run_name}")
    writer.add_config(vars(args))
    # TRY NOT TO MODIFY
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    local_obs_space = env.observation_space(env.possible_agents[0])
    global_obs_space = env.state_space
    action_space = env.action_space(env.possible_agents[0])[args.control]
    partial_obs_extractor = DfacSpaceExtractor_max(local_obs_space, global_obs_space, args.control) if args.task == 'max' else DfacSpaceExtractor_track(local_obs_space, global_obs_space, args.control, track_power)
    partial_obs_space = partial_obs_extractor.observation_space
    features_extractor_params = {
        "order":args.fourier_order,
        "hyper": args.fourier_hyper,
        "learnable": args.fourier_learnable,
        "max_dim": args.fourier_maxdim,
        "seed": args.seed,
    }
    hidden_layer_nn = [] if not isinstance(args.hidden_layer_nn, tuple) else args.hidden_layer_nn

    agents = [
        AGENT_TYPE_DICT[args.agent_type](partial_obs_space, action_space, hidden_layer_nn, features_extractor_params).to(device)
        for _ in range(args.num_agents)
    ]
    optimizers = [
        optim.RMSprop(agent.parameters(), alpha=0.99, lr=args.learning_rate, eps=1e-5, weight_decay=0)
        for agent in agents
    ]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs, args.num_agents) + partial_obs_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs, args.num_agents) + action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs, args.num_agents)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # set single wind conditions for reproducibility
    env.reset(options={"wind_speed": 8, "wind_direction": 270})

    for step in range(1, args.total_timesteps):

        global_step += args.num_envs
        if global_step % 50 == 0:
            print(f'step : {global_step}')

        # ALGO LOGIC: action logic
        powers = []
        with torch.no_grad():
            for idagent, agent in enumerate(agents):
                last_obs, reward, terminations, truncations, infos = env.last()
                global_obs = env.state()
                last_obs = torch.Tensor(partial_obs_extractor(last_obs, global_obs, step)).to(device)
                action, logprob, _, value = agent.get_action_and_value(last_obs)
                last_done = np.logical_or(terminations, truncations)
                last_done = torch.Tensor([last_done.astype(int)]).to(device)

                # store values
                values[step, :, idagent] = value.flatten()
                logprobs[step, :, idagent] = logprob
                obs[step, :, idagent] = last_obs
                dones[step-1, :, idagent] = last_done
                rewards[step-1, :, idagent] = torch.tensor(reward).to(device).view(-1)
                if "power" in infos:
                    powers.append(infos["power"])
                    writer.add_scalar(f"farm/power_T{idagent}", infos["power"], global_step)
                if "load" in infos:
                    writer.add_scalar(f"farm/load_T{idagent}", sum(np.abs(infos["load"])), global_step)
                writer.add_scalar(f"farm/controls/yaw_T{idagent}", last_obs[0].item(), global_step)

                # store next action
                env.step(make_action(action))
                actions[step, :, idagent] = action
        
        writer.add_scalar(f"farm/power_total", sum(powers), global_step)
        writer.add_scalar(f"farm/reward", float(reward[0]), global_step)

        # bootstrap value if not done
        # indice to learn from
        indice = step-1

        # available data to train on
        if indice > 1:
            advantages = torch.zeros_like(rewards[indice]).to(device)
            with torch.no_grad():
                for idagent, agent in enumerate(agents):
                    next_value = agent.get_value(obs[indice+1, :, idagent]).reshape(1, -1)
                    nextnonterminal = 1.0 - dones[indice, :, idagent]
                    delta = rewards[indice, :, idagent] + args.gamma * next_value * nextnonterminal - values[indice, :, idagent]
                    advantages[:, idagent] = delta
            returns = advantages + values[indice]

            if args.norm_adv:
                norm_advantages = (advantages - advantages.mean(0)) / (advantages.std(0) + 1e-8)
            
            # one step update
            for idagent, agent in enumerate(agents):
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    obs[indice, :, idagent], actions.long()[indice, :, idagent]
                )

                # Policy loss
                pg_loss = -(advantages[:, idagent] * newlogprob).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                v_loss = F.mse_loss(returns[:, idagent], newvalue)
                # v_loss = 0.5 * ((newvalue - returns[step, :, idagent]) ** 2).mean()

                # entropy_loss = entropy.mean()
                loss = pg_loss + v_loss * args.vf_coef

                optimizers[idagent].zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizers[idagent].step()

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                writer.add_scalar(f"charts/agent_{idagent}/learning_rate", optimizers[idagent].param_groups[0]["lr"], global_step)
                writer.add_scalar(f"losses/agent_{idagent}/value_loss", v_loss.item(), global_step)
                writer.add_scalar(f"losses/agent_{idagent}/policy_loss", pg_loss.item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))

            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        for idagent, agent in enumerate(agents):
            torch.save(agent.state_dict(), model_path+f"_{idagent}")
        print(f"model saved to {model_path}")

    env.close()
    writer.close()

    # env.mdp.interface.render(out_dir=f"runs/{run_name}/")

    # Prepare plots
    fig = plot_env_history(env)
    fig.savefig(f"runs/{run_name}/plot.png")

    print("stop")








