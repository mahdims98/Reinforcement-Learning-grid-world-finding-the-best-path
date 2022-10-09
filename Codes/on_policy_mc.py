from grid_world_env import GridworldEnv
import numpy as np
from scipy.optimize import fsolve
import time


class OnPolicyAgent:
    def __init__(self, env: GridworldEnv, max_epsilon, min_epsilon, gamma):
        self.env = env
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.policy = np.full((self.env.nS, self.env.nA), 1 / self.env.nA)
        self.Q = np.full((self.env.nS, self.env.nA), 0, dtype=np.float32)
        self.state_action_return = np.full((self.env.nS, self.env.nA), 0)
        self.state_action_return_counter = np.full((self.env.nS, self.env.nA), 0)
        self.possible_actions = np.arange(self.env.nA)
        self.gamma = gamma

    def learn(self, num_of_episodes_to_be_generated, episode_max_length=None):
        epsilon_decay = self.calculate_epsilon_decay(max_epsilon=self.epsilon, min_epsilon=self.min_epsilon,
                                                     num_of_episodes=num_of_episodes_to_be_generated)
        episode_reward_history = [0] * num_of_episodes_to_be_generated
        episode_time = []

        for i in range(num_of_episodes_to_be_generated):
            start_time = time.time()
            episode_actions, episode_rewards, episode_states = self.generate_episode(policy=self.policy)

            G = 0
            episode_reward_history[i] = sum(episode_rewards)
            if episode_max_length is not None and len(episode_states) > episode_max_length:
                continue

            for time_step in np.arange(len(episode_states) - 1, -1, -1):
                G = self.gamma * G + episode_rewards[time_step]
                state = episode_states[time_step]
                action = episode_actions[time_step]
                if not (state, action) in zip(episode_states[:time_step], episode_actions[:time_step]):

                    self.state_action_return[state][action] += G
                    self.state_action_return_counter[state][action] += 1
                    self.Q[state][action] = self.state_action_return[state, action] / self.state_action_return_counter[
                        state, action]
                    best_action = np.argmax(self.Q[state])
                    self.policy[state, :] = self.epsilon / self.env.nA
                    self.policy[state][best_action] = 1 - self.epsilon + self.epsilon / self.env.nA
                else:
                    continue

            self.epsilon = self.epsilon * epsilon_decay
            print("Episode number", i, " Generated")
            end = time.time()
            episode_time.append(end - start_time)

        time_per_episode = np.mean(episode_time)
        return episode_reward_history, self.policy, time_per_episode

    def generate_episode(self, policy):

        self.env.reset()
        episode_states = []
        episode_actions = []
        episode_rewards = []
        done = False
        state = int(np.where(self.env.isd == 1)[0])  # initial state
        # Generating an episode
        while not done:
            episode_states.append(state)
            action = np.random.choice(self.possible_actions, p=policy[state].flatten())
            state, reward, done, info = self.env.step(action)
            episode_actions.append(action)
            episode_rewards.append(reward)

        # take another action in the final state

        episode_states.append(state)
        action = np.random.choice(self.possible_actions, p=policy[state].flatten())
        state, reward, done, info = self.env.step(action)
        episode_actions.append(action)
        episode_rewards.append(reward)

        return episode_actions, episode_rewards, episode_states

    @staticmethod
    def calculate_epsilon_decay(max_epsilon, min_epsilon, num_of_episodes):
        if max_epsilon == min_epsilon:
            return 1
        else:
            def function(x):
                return max_epsilon * (x ** num_of_episodes) - min_epsilon

            guess = np.array([0.9999])
            return fsolve(function, guess)
