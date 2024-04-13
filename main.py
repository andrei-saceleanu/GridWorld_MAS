import numpy as np
import matplotlib.pyplot as plt

from algorithms import *
from gridworld import GridWorld

def plot_for_env(env_name, q_tr, q_te, sarsa_tr, sarsa_te):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(env_name)
    ax[0].plot(list(range(len(q_tr))), q_tr, label="Q-Learning train")
    ax[0].plot(list(range(len(sarsa_tr))), sarsa_tr, label="SARSA train")
    ax[0].legend(loc="lower right")
    ax[1].plot(list(range(len(q_te))), q_te, label="Q-Learning test")
    ax[1].plot(list(range(len(sarsa_te))), sarsa_te, label="SARSA test")
    ax[1].legend(loc="lower right")
    plt.savefig(f"{env_name}.png")
    plt.close()

def main():

    cfg = {
        "num_actions": 4,
        "world_size": [10, 7],
        "world_type": "A"
    }
    env = GridWorld(cfg=cfg)

    random.seed(0)
    names = ["Q-Learning", "SARSA"]

    gamma = 0.99
    epsilon = 0.3
    alpha = 0.1
    for name in names:
        if name == "Q-Learning":
            q_tr, q_te, qlens = q_learning(env, gamma, epsilon, alpha, 10000, 50)
        else:
            sarsa_tr, sarsa_te, slens = sarsa(env, gamma, epsilon, alpha, 10000, 50)

    plot_for_env("GridWorld_A", q_tr, q_te, sarsa_tr, sarsa_te)
    plt.figure()
    plt.plot(list(range(len(qlens))), qlens, label="Q lengths")
    plt.plot(list(range(len(slens))), slens, label="SARSA lengths")
    plt.legend()
    plt.savefig("ep_lens.png")
    plt.close()

if __name__=="__main__":
    main()