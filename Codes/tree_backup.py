from grid_world_env import GridworldEnv
import numpy as np
from scipy.optimize import fsolve
import time


class TreeBackupAgent:

    def __init__(self, env: GridworldEnv, max_epsilon, min_epsilon, max_alpha, min_alpha, gamma, step_length):
        self.env = env
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.gamma = gamma
        self.alpha = max_alpha
        self.min_alpha = min_alpha
        self.Q = np.full((self.env.nS, self.env.nA), 0, dtype=np.float32)
        self.possible_actions = np.arange(self.env.nA)
        self.policy = np.full((self.env.nS, self.env.nA), 1 / self.env.nA)
        self.policy_b = np.full((self.env.nS, self.env.nA), 1 / self.env.nA)
        self.step_length = step_length

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

            done_counter = 0
            T = np.inf
            t = 0
            tau = 0
            episode_states = []
            episode_actions = []
            episode_rewards = []
            state = int(np.where(self.env.isd == 1)[0])
            action = np.random.choice(self.possible_actions, p=self.policy[state].flatten())
            episode_states.append(state)
            episode_rewards.append(0)
            episode_actions.append(action)
            while tau != T - 1:
                if t < T:
                    next_state, next_reward, done, info = self.env.step(action)
                    episode_rewards.append(next_reward)
                    episode_states.append(next_state)
                    if done:
                        done_counter += 1

                    if done_counter == 2:
                        T = t + 1

                    else:
                        next_action = np.random.choice(self.possible_actions, p=self.policy[next_state].flatten())
                        episode_actions.append(next_action)

                tau = t + 1 - self.step_length
                if tau >= 0:
                    if t + 1 >= T:
                        G = next_reward
                    else:
                        G = episode_rewards[t + 1] + self.gamma * sum(
                            self.policy[next_state][j] * self.Q[next_state][j] for j in self.possible_actions)

                    for k in np.arange(min(t, T - 1), tau, -1):
                        boot_strap = 0
                        for a in self.possible_actions:
                            if a != episode_actions[k]:
                                boot_strap += self.policy[episode_states[k]][a] * self.Q[episode_states[k]][
                                    a] + self.gamma * self.policy[episode_states[k]][episode_actions[k]] * G
                        G = episode_rewards[k] + boot_strap

                    self.Q[episode_states[tau]][episode_actions[tau]] = self.Q[episode_states[tau]][
                                                                            episode_actions[tau]] + self.alpha * (
                                                                                G - self.Q[episode_states[tau]][
                                                                            episode_actions[tau]])

                    best_action = np.argmax(self.Q[state])
                    self.policy[state, :] = self.epsilon / self.env.nA
                    self.policy[state][best_action] = 1 - self.epsilon + self.epsilon / self.env.nA

                action = next_action
                state = next_state

                t += 1

            episode_reward_history[i] = sum(episode_rewards)
            self.epsilon = self.epsilon * epsilon_decay
            print("Episode number", i, " Generated")
            end = time.time()
            episode_time.append(end - start_time)
            # print(self.epsilon, epsilon_decay)

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
