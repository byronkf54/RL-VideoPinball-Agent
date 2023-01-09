# Import Libraries
import numpy as np
import random
from collections import deque
import torch
import gym

# allows agent to run on multiple environments at one time
from worker_wrapper import SubprocVecEnv
# source code from stable baselines
from stable_baselines_code import make_env

# custom files
from agents import DQNAgent, RainbowAgent
from config import *


def run_random_policy():
    state = random_policy_env.reset()
    score = 0
    episode = 1
    print(f"Running random policy to fill Replay Buffer to size: {PRE_FILL_BUFFER}")
    for i in range(PRE_FILL_BUFFER):
        action = np.random.randint(action_size)
        next_state, reward, done, _ = random_policy_env.step(action)
        agent.buffer.add(state, action, reward, next_state, done)
        score += reward
        if done:
            print(f"Episode {episode} Score {score}")
            episode += 1
            score = 0
            state = random_policy_env.reset()


def run(frames, eps_frames, min_eps):
    ma_scores = deque(maxlen=100)  # store the last 100 scores (used to find 100 MA)
    if eps_fixed:
        eps = 0
    else:
        eps = 1
    eps_start = 1
    d_eps = eps_start - min_eps
    episode = 1
    state = env.reset()
    score = 0
    print("Running Agent")
    for frame_count in range(1, frames + 1):
        # env.render()
        action = agent.act(state, eps)
        next_state, reward, done, _ = env.step(action)
        for s, a, r, ns, d in zip(state, action, reward, next_state, done):
            agent.step(s, a, r, ns, d)
        state = next_state
        score += reward
        # only decay epsilon if not fixed
        if not eps_fixed:
            eps = max(eps_start - ((frame_count * d_eps) / eps_frames), min_eps)

        if done.any():
            ma_scores.append(score)  # save most recent score
            print(f'Episode {episode} Frame {frame_count} Average Score: {np.mean(ma_scores)} Score: {int(score)}')
            if SAVE_MODEL and episode % 10 == 0:
                torch.save(agent.qnetwork_local.state_dict(), f"./rainbow/{episode}" + ".pth")
            episode += 1
            state = env.reset()
            score = 0

    return np.mean(ma_scores)


class AgentOptions:
    def __init__(self, dqn, dueling, c51, per, noisy, rainbow):
        self.choices = ["dqn", "dqn_per", "dqn_noisy", "dqn_per_noisy", "dueling", "dueling_per", "dueling_noisy",
                        "dueling_per_noisy", "dueling_c51_per_noisy", "rainbow"]
        self.dqn = dqn
        self.dueling = dueling
        self.c51 = c51
        self.per = per
        self.noisy = noisy
        self.rainbow = rainbow

        if rainbow:
            self.dqn = False
            self.dueling = True
            self.c51 = True
            self.per = True
            self.noisy = True
            self.agent = "dueling_c51_per_noisy"  # all features combined
        else:
            self.agent = ""
            for attr, value in self.__dict__.items():
                if value:
                    self.agent += attr + "_"
            self.agent = self.agent[:-7]

        self.error = False

        if self.agent not in self.choices:
            print(agent_options.agent)
            print("ERROR invalid choice of agent, please consult the list below for options")
            print(agent_options.choices)
            exit()

    def stringify_options(self):
        options_text = "Running agent with "
        if self.rainbow:
            return options_text + "Rainbow activated"
        for attr, value in self.__dict__.items():
            if value:
                options_text += attr + " "
        return options_text + "activated"


if __name__ == "__main__":
    agent_options = AgentOptions(dqn=False, dueling=True, c51=True, per=True, noisy=True, rainbow=True)

    options = agent_options.stringify_options()
    print(options)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)

    env = SubprocVecEnv([lambda: make_env(gym.make(ENV_NAME)) for _ in range(1)])
    random_policy_env = make_env(gym.make(ENV_NAME))
    env.seed(SEED)

    action_size = env.action_space.n
    state_size = env.observation_space.shape

    if agent_options.rainbow:
        agent = RainbowAgent(input_dim=state_size,
                             output_dim=action_size,
                             agent=agent_options,
                             n_step=N_STEP,
                             batch_size=BATCH_SIZE,
                             buffer_size=BUFFER_SIZE,
                             lr=LR,
                             tau=TAU,
                             discount_factor=DISCOUNT_FACTOR,
                             device=device,
                             seed=SEED,
                             atom_size=ATOM_SIZE,
                             Vmax=VMAX,
                             Vmin=VMIN)
    else:
        agent = DQNAgent(input_dim=state_size,
                         output_dim=action_size,
                         agent=agent_options,
                         n_step=N_STEP,
                         batch_size=BATCH_SIZE,
                         buffer_size=BUFFER_SIZE,
                         lr=LR,
                         tau=TAU,
                         discount_factor=DISCOUNT_FACTOR,
                         device=device,
                         seed=SEED)

    # adding frames from random policy to the buffer before training begins
    if PRE_FILL_BUFFER > 0:
        run_random_policy()
        print("Buffer size: ", agent.buffer.__len__())

    # set epsilon frames to 0 so no epsilon exploration
    if agent_options.noisy:
        eps_fixed = True
    else:
        eps_fixed = False

    final_average100 = run(frames=FRAMES, eps_frames=EPSILON_FRAMES, min_eps=MIN_EPSILON)
