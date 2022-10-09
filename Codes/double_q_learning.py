from grid_world_env import GridworldEnv
import numpy as np
from scipy.optimize import fsolve
import time

class DoubleQLearningAgent:

    def __init__(self, env: GridworldEnv, max_epsilon, min_epsilon, max_alpha, min_alpha, gamma):
        self.env = env
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.alpha = max_alpha
        self.min_alpha = min_alpha
        self.Q_1 = np.full((self.env.nS, self.env.nA), 0, dtype=np.float32)
        self.Q_2 = np.full((self.env.nS, self.env.nA), 0, dtype=np.float32)
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

            while not done:
                next_state, reward, done, info = self.env.step(action)
                episode_rewards.append(reward)
                next_action = np.random.choice(self.possible_actions, p=self.policy[next_state].flatten())
                update_which_Q = np.random.choice([1, 2], p=[0.5, 0.5])

                if update_which_Q == 1:
                    arg_max_Q_1 = np.argmax(self.Q_1[next_state])
                    self.Q_1[state][action] = self.Q_1[state][action] + self.alpha * (
                                reward + self.gamma * self.Q_2[next_state][arg_max_Q_1] - self.Q_1[state][action])
                else:
                    arg_max_Q_2 = np.argmax(self.Q_2[next_state])
                    self.Q_2[state][action] = self.Q_2[state][action] + self.alpha * (
                                reward + self.gamma * self.Q_1[next_state][arg_max_Q_2] - self.Q_2[state][action])

                self.update_policy(state)
                state = next_state
                action = next_action

            # repeat for the last step

            next_state, reward, done, info = self.env.step(action)
            episode_rewards.append(reward)
            next_action = np.random.choice(self.possible_actions, p=self.policy[next_state].flatten())
            update_which_Q = np.random.choice([1, 2], p=[0.5, 0.5])
            if update_which_Q == 1:
                arg_max_Q_1 = np.argmax(self.Q_1[next_state])
                self.Q_1[state][action] = self.Q_1[state][action] + self.alpha * (
                        reward + self.gamma * self.Q_2[next_state][arg_max_Q_1] - self.Q_1[state][action])

            if update_which_Q == 2:
                arg_max_Q_2 = np.argmax(self.Q_2[next_state])
                self.Q_2[state][action] = self.Q_2[state][action] + self.alpha * (
                        reward + self.gamma * self.Q_1[next_state][arg_max_Q_2] - self.Q_2[state][action])

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
        best_action = np.argmax(self.Q_1[state] + self.Q_2[state])
        self.policy[state, :] = self.epsilon / self.env.nA
        self.policy[state][best_action] = 1 - self.epsilon + self.epsilon / self.env.nA
