from grid_world_env import GridworldEnv
from on_policy_mc import OnPolicyAgent
from off_policy_mc import OffPolicyAgent
from sarsa import SarsaAgent
from tree_backup import TreeBackupAgent
from expected_sarsa_two_step import ExpectedSarsaAgent
from double_q_learning import DoubleQLearningAgent
from grid_world_env_better import GridworldEnvBetter
from matplotlib import pyplot as plt
from evaluator_mdp import Evaluator
import numpy as np
import seaborn as sns

env = GridworldEnvBetter()  # can be changed to GridworldEnv()
env.reset()
env._render()

double_q_learning_agent_3 = DoubleQLearningAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.2, min_alpha=0.2, gamma=1)
expected_sarsa_agent_3 = ExpectedSarsaAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
on_policy_agent_3 = OnPolicyAgent(env=env, max_epsilon=0.5, min_epsilon=0.001, gamma=1)
sarsa_agent_2 = SarsaAgent(env=env, max_epsilon=0.5, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
tree_backup_agent_1 = TreeBackupAgent(env=env, max_epsilon=0.999, min_epsilon=0.01, max_alpha=0.5, min_alpha=0.5, gamma=1, step_length=7)
off_policy_agent_3 = OffPolicyAgent(env=env, max_epsilon_b=0.999, min_epsilon_b=0.001, gamma=1)
agent_dict = {"Double Q": double_q_learning_agent_3, "Expected Sarsa": expected_sarsa_agent_3,"On policy" :on_policy_agent_3, "Sarsa": sarsa_agent_2, "Tree backup": tree_backup_agent_1,"Off policy" :off_policy_agent_3 }
evaluator = Evaluator()
evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=250, episode_max_length=2000)


# on_policy_agent_1 = OnPolicyAgent(env=env, max_epsilon=0.99, min_epsilon=0.001, gamma=1)
# on_policy_agent_2 = OnPolicyAgent(env=env, max_epsilon=0.4, min_epsilon=0.4, gamma=1)
# on_policy_agent_3 = OnPolicyAgent(env=env, max_epsilon=0.5, min_epsilon=0.001, gamma=1)
# agent_dict = {"On policy γ=1, epsilon: 0.99 to 0.001": on_policy_agent_1, "On policy γ=1, epsilon=0.5":on_policy_agent_2, "On policy γ=1, epsilon: 0.5 to 0.001" :on_policy_agent_3}

# off_policy_agent_1 = OffPolicyAgent(env=env, max_epsilon_b=0.9, min_epsilon_b=0.001, gamma=0.5)
# off_policy_agent_2 = OffPolicyAgent(env=env, max_epsilon_b=0.9, min_epsilon_b=0.3, gamma=0.5)
# off_policy_agent_3 = OffPolicyAgent(env=env, max_epsilon_b=0.5, min_epsilon_b=0.001, gamma=0.5)
# agent_dict = {"Off policy γ=0.5, epsilon for b: 0.9 to 0.001": off_policy_agent_1, "Off policy γ=0.5, epsilon for b: 0.9 to 0.3":off_policy_agent_2, "Off policy γ=0.5, epsilon: 0.5 to 0.001" :off_policy_agent_3}
# evaluator = Evaluator()
# evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=500, episode_max_length=3500)

# double_q_learning_agent_1 = DoubleQLearningAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
# double_q_learning_agent_2 = DoubleQLearningAgent(env=env, max_epsilon=0.5, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
# double_q_learning_agent_3 = DoubleQLearningAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.2, min_alpha=0.2, gamma=1)
# agent_dict = {"Double Q γ=1, ε: 0.999 to 0.001 α:0.5": double_q_learning_agent_1, "Double Q γ=1, ε: 0.5 to 0.001 α:0.5": double_q_learning_agent_2, "Double Q γ=1 ε: 0.999 to 0.001 α:0.2": double_q_learning_agent_3}
# evaluator = Evaluator()
# evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=500, episode_max_length=3500)

# tree_backup_agent_1 = TreeBackupAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1, step_length=3)
# tree_backup_agent_2 = TreeBackupAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1, step_length=5)
# tree_backup_agent_3 = TreeBackupAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1, step_length=7)
# agent_dict = {"Tree backup γ=1, ε: 0.999 to 0.001 α:0.5 n:3": tree_backup_agent_1, "Double Q γ=1, ε: 0.999 to 0.001 α:0.5 n:5": tree_backup_agent_2, "Double Q γ=1 ε: 0.999 to 0.001 α:0.5 n:7": tree_backup_agent_3}
# evaluator = Evaluator()
# evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=500, episode_max_length=3500)



# sarsa_agent_1 = SarsaAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0., min_alpha=0.5, gamma=1)
# sarsa_agent_2 = SarsaAgent(env=env, max_epsilon=0.5, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
# sarsa_agent_3 = SarsaAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.3, min_alpha=0.3, gamma=1)
# agent_dict = {"Sarsa γ=1, ε: 0.5 to 0.001 α:0.5": sarsa_agent_1, "Sarsa γ=1, ε: 0.999 to 0.001 α:0.5": sarsa_agent_2, "Sarsa γ=1 ε: 0.999 to 0.001 α:0.3": sarsa_agent_3}
# evaluator = Evaluator()
# evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=500, episode_max_length=3500)


# expected_sarsa_agent_1 = ExpectedSarsaAgent(env=env, max_epsilon=0.8, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
# expected_sarsa_agent_2 = ExpectedSarsaAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.5, min_alpha=0.5, gamma=1)
# expected_sarsa_agent_3 = ExpectedSarsaAgent(env=env, max_epsilon=0.999, min_epsilon=0.001, max_alpha=0.4, min_alpha=0.4, gamma=1)
# agent_dict = {"Expected Sarsa γ=1, ε: 0.8 to 0.001 α:0.5": expected_sarsa_agent_1, "Expected Sarsa γ=1, ε: 0.999 to 0.001 α:0.3": expected_sarsa_agent_2, "Expected Sarsa γ=1 ε: 0.999 to 0.001 α:0.4": expected_sarsa_agent_3}
# evaluator = Evaluator()
# evaluator.compare_rewards_per_episode(agent_dict, num_of_episodes_to_be_generated=150, episode_max_length=3500)








