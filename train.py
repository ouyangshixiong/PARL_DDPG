#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import time
import parl
import os.path

from env import ContinuousCartPoleEnv
from model_2_fc import Model
# from mujoco_model import MujocoModel as Model
from agent import Agent
from parl.utils import logger, action_mapping, ReplayMemory

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3
GAMMA = 0.99
TAU = 0.001
MEMORY_SIZE = int(1e6)
MEMORY_WARMUP_SIZE = 1e4
BATCH_SIZE = 128
REWARD_SCALE = 0.1
ENV_SEED = 1


def run_train_episode(env, agent, rpm):
    obs = env.reset()
    total_reward = 0
    steps = 0
    while True:
        steps += 1
        batch_obs = np.expand_dims(obs, axis=0)
        action = agent.predict(batch_obs.astype('float32'))

        # Add exploration noise, and clip to [-1.0, 1.0]
        action = np.clip(np.random.normal(action, 1.0), -1.0, 1.0)
        action = action_mapping(action, env.action_space.low[0],
                                env.action_space.high[0])

        next_obs, reward, done, info = env.step(action)

        rpm.append(obs, action, REWARD_SCALE * reward, next_obs, done)

        if rpm.size() > MEMORY_WARMUP_SIZE and (steps % 5) == 0:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

        obs = next_obs
        total_reward += reward

        if done or steps >= 200:
            break
    return total_reward


def run_evaluate_episode(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        total_reward = 0
        steps = 0
        while True:
            batch_obs = np.expand_dims(obs, axis=0)
            action = agent.predict(batch_obs.astype('float32'))
            action = action_mapping(action, env.action_space.low[0],
                                    env.action_space.high[0])
            steps += 1
            next_obs, reward, done, info = env.step(action)

            obs = next_obs
            total_reward += reward
            if render:
                env.render()
            if done or steps >= 200:
                break
        eval_reward.append(total_reward)
    return np.mean(eval_reward)


def main():
    env = ContinuousCartPoleEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = Model(act_dim)
    algorithm = parl.algorithms.DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, act_dim)

    while rpm.size() < MEMORY_WARMUP_SIZE:
        run_train_episode(env, agent, rpm)

    episode = 0
    while episode < 30000:
        for i in range(50):
            train_reward = run_train_episode(env, agent, rpm)
            episode += 1
            # logger.info('Episode: {} Reward: {}'.format(episode, train_reward))

        evaluate_reward = run_evaluate_episode(env, agent, False)
        logger.info('Episode {}, Evaluate reward: {}'.format(
            episode, evaluate_reward))
        if( evaluate_reward == 200 ):
            break
    agent.save('./model_dir')

def test():
    env = ContinuousCartPoleEnv()

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    model = Model(act_dim)
    algorithm = parl.algorithms.DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Agent(algorithm, obs_dim, act_dim)

    if os.path.exists('./model_dir'):
        agent.restore('./model_dir')

    eval_reward = run_evaluate_episode(env, agent, True)
    logger.info('test_reward:{}'.format(
            eval_reward))

if __name__ == '__main__':
    main()
    test()
