from grid_world_env import GridworldEnv
import numpy as np
from scipy.optimize import fsolve
import time

class SarsaAgent:

    def __init__(self, env: GridworldEnv, max_epsilon, min_epsilon, max_alpha, min_alpha, gamma):
        self.env = env
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.alpha = max_alpha
        self.min_alpha = min_alpha
        self.Q = np.full((self.env.nS, self.env.nA), 0, dtype=np.float32)
        self.possible_actions = np.arange(self.env.nA)
        self.policy = np.full((self.env.nS, self.env.nA), 1 / self.env.nA)

    def learn(self, num_of_episodes_to_be_generated, episode_max_length=None):

        epsilon_decay = self.calculate_epsilon_decay(max_epsilon=self.epsilon,
                                                     min_epsilon=self.min_epsilon,
                                                     num_of_episodes=num_of_episodes_to_be_generated)

        episode_reward_history = [0] * num_of_episodes_to_be_generated
        episode_time = []

        for i in range(num_of_episodes_to_be_generated):
            start_time = time.time()
            self.env.reset()
            episode_rewards = []
            done = False

            state = int(np.where(self.env.isd == 1)[0])  # initial state
            action = np.random.choice(self.possible_actions, p=self.policy[state].flatten())
            episode_length = 0
            while not done:
                next_state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)
                next_action = np.random.choice(self.possible_actions, p=self.policy[next_state].flatten())

                self.Q[state][action] = self.Q[state][action] + self.alpha * (
                            reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
                self.update_policy(state)
                state = next_state
                action = next_action
                episode_length +=1
                if episode_length > episode_max_length:
                    break

            if episode_length > episode_max_length:
                continue

            # for the last step in final state
            # env = GridworldEnv()

            # next_state, reward, done, info = env.step(action)
            next_state, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)
            next_action = np.random.choice(self.possible_actions, p=self.policy[next_state].flatten())
            self.Q[state][action] = self.Q[state][action] + self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - self.Q[state][action])
            self.update_policy(state)
            state = next_state
            action = next_action

            self.epsilon *= epsilon_decay
            episode_reward_history[i] = sum(episode_rewards)
            print("Episode number", i, " Generated")
            end = time.time()
            episode_time.append(end - start_time)

        time_per_episode = np.mean(episode_time)
        return episode_reward_history, self.policy, time_per_episode

    @staticmethod
    def calculate_epsilon_decay(max_epsilon, min_epsilon, num_of_episodes):
        if max_epsilon == min_epsilon:
            return 1
        else:
            def function(x):
                return max_epsilon * (x ** num_of_episodes) - min_epsilon

            guess = np.array([0.9999])
            return fsolve(function, guess)

    def update_policy(self, state):
        best_action = np.argmax(self.Q[state])
        self.policy[state, :] = self.epsilon / self.env.nA
        self.policy[state][best_action] = 1 - self.epsilon + self.epsilon / self.env.nA
