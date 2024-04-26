import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class Visuals():

    """This class takes in metrics from the Bandit algorithm and allows you to make visuals"""

    def __init__(self, bandit_reward, original_reward, testing_rounds):
        self.bandit_reward = bandit_reward
        self.original_reward = original_reward
        self.testing_rounds = testing_rounds

        self.score_dict = {"Original Reward": self.original_reward,
                           "Bandit Reward": self.bandit_reward}

    def bar_comparision(self):
        sns.barplot(x=self.score_dict.keys(), y=self.score_dict.values())
        plt.title("Comparison of Bandit Reward vs Original Reward")
        plt.xlabel("Policies")
        plt.ylabel("Average Reward")

        offset = (self.bandit_reward - self.original_reward)
        plt.ylim(self.original_reward - 2*offset, self.bandit_reward + 2*offset)
        
        plt.yticks(np.linspace(self.original_reward - 2*offset, self.bandit_reward + 2*offset, 10))

        plt.axhline(self.score_dict["Bandit Reward"], color='red')

        plt.show()

    def table(self):
        from prettytable import PrettyTable 

        table = PrettyTable(['Bandit Policy', 'Original Policy'])

        PrettyTable.add_row([self.bandit_reward, self.original_reward])
        PrettyTable.add_row([self.testing_rounds, self.testing_rounds])
        PrettyTable.add_row(["Diverse Arm Scores", "Equal Arm Scores"])

        print(table)



if __name__ == '__main__':
    viz = Visuals(bandit_reward=0.08233069306930693, original_reward=0.08099801980198021, testing_rounds=5000000)
    viz.bar_comparision()
