from grid_world_env import GridworldEnv
import numpy as np
from scipy.optimize import fsolve
import time

class ExpectedSarsaAgent:

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

            state_step_0 = int(np.where(self.env.isd == 1)[0])  # initial state
            action_step_0 = np.random.choice(self.possible_actions, p=self.policy[state_step_0].flatten())
            episode_length_counter = 0
            expected_reward_step_1 = 0
            remember_its_done = False
            while not done:
                state_step_1, reward_step_0, done_step_0, info = self.env.step(action_step_0)
                episode_rewards.append(reward_step_0)

                if remember_its_done:
                    # terminal state
                    for action_step_1 in self.possible_actions:
                        expected_reward_step_1 += self.policy[state_step_1, action_step_1] * self.Q[state_step_1][
                            action_step_1]
                    self.Q[state_step_0][action_step_0] = self.Q[state_step_0][action_step_0] + self.alpha * (
                            reward_step_0 + self.gamma * expected_reward_step_1 - self.Q[state_step_0][action_step_0])
                    self.update_policy(state_step_0)
                    break

                action_step_1 = np.random.choice(self.possible_actions, p=self.policy[state_step_1].flatten())
                state_step_2, reward_step_1, done_step_1, info = self.env.step(action_step_1)
                episode_rewards.append(reward_step_1)
                expected_reward_step_2 = 0

                if not done_step_0 and not done_step_1:
                    done = False
                elif done_step_0:
                    done = True
                elif done_step_1:
                    done = False
                    remember_its_done = True

                for action_step_2 in self.possible_actions:
                    expected_reward_step_2 += self.policy[state_step_2, action_step_2] * self.Q[state_step_2][
                        action_step_2]

                self.Q[state_step_0][action_step_0] = self.Q[state_step_0][action_step_0] + self.alpha * (
                            reward_step_0 + self.gamma * reward_step_1 + self.gamma ** 2 * expected_reward_step_2 -
                            self.Q[state_step_0][action_step_0])
                self.update_policy(state_step_0)
                state_step_0 = state_step_2
                action_step_0 = np.random.choice(self.possible_actions, p=self.policy[state_step_0].flatten())
                episode_length_counter += 2

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
