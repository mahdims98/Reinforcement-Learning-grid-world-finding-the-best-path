import numpy as np
from amalearn.agent import AgentBase
from matplotlib import pyplot as plt
import seaborn as sns
import io
from colorama import Fore, Back, Style
import sys

class Evaluator:
    def __init__(self):
        CB91_Blue = '#2CBDFE'
        CB91_Green = '#47DBCD'
        CB91_Pink = '#F3A0F2'
        CB91_Purple = '#9D2EC5'
        CB91_Violet = '#661D98'
        CB91_Amber = '#F5B14C'
        self.color_list = [CB91_Blue, CB91_Amber, CB91_Green, CB91_Purple, CB91_Pink, CB91_Violet]

    def convert_policy(self, policy,mode='human', close=False):
        policy_char_format = np.empty((policy.shape[0], 1), dtype=object)

        outfile = io.StringIO() if mode == 'ansi' else sys.stdout

        grid = np.arange(100).reshape((10,10))
        it = np.nditer(grid, flags=['multi_index'])

        while not it.finished:
            s = it.iterindex
            y, x = it.multi_index


            best_action = np.argmax(policy[s, :])
            if best_action == 0:
                policy_char_format = "  ↑  "
            elif best_action == 1:
                policy_char_format = "  →  "
            elif best_action == 2:
                policy_char_format = "  ↓  "
            else:
                policy_char_format = "  ←  "

            if s == 0:
                output = Back.GREEN + Style.BRIGHT + Fore.WHITE + "     " + Style.RESET_ALL
            elif s in [13, 62, 41]:
                output = Back.BLUE + "     " + Style.RESET_ALL
            elif s in [43, 33, 53, 34, 35, 54]:
                output = Back.CYAN + Style.BRIGHT + Fore.BLACK + "  █  " + Style.RESET_ALL
            else:

                output = Back.CYAN + Style.BRIGHT + Fore.BLACK + policy_char_format + Style.RESET_ALL



            if x == 0:
                output = output.lstrip()
            if x == 10 - 1:
                output = output.rstrip()

            outfile.write(output)

            if x == 10 - 1:
                outfile.write("\n")
            it.iternext()

        return policy_char_format


    def compare_rewards_per_episode(self, agent_dict: dict, num_of_episodes_to_be_generated, episode_max_length):
        fig, ax = plt.subplots()
        fig2, ax2 = plt.subplots()
        sns.set_style()
        time_per_episode_agents = dict()
        i = 0
        for agent_name in agent_dict:
            episode_reward_history, policy, time_per_episode = agent_dict[agent_name].learn(num_of_episodes_to_be_generated=num_of_episodes_to_be_generated, episode_max_length=episode_max_length)

            ax2.bar(agent_name, time_per_episode, color=self.color_list[i])

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            ax2.set_ylabel("Time spent per Episode")
            ax2.set_xlabel("")
            ax2.grid(False)


            ax.plot(episode_reward_history, label=agent_name, color=self.color_list[i])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_ylabel("Reward per Episode")
            ax.set_xlabel("Episode Number")
            ax.grid(True, axis='y')
            ax.legend()
            i += 1
            if i > 5:
                i = 0
            print(agent_name)
            print(self.convert_policy(policy))

        plt.show()



















