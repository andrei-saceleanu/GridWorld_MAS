import numpy as np
import matplotlib.pyplot as plt

from algorithms import *
from gridworld import GridWorld

def plot_for_env(env_name, q_tr, q_te, sarsa_tr, sarsa_te, dq_tr, dq_te):
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True)
    fig.suptitle(env_name)
    ax[0].plot(list(range(len(q_tr))), q_tr, label="Q-Learning train")
    ax[0].plot(list(range(len(sarsa_tr))), sarsa_tr, label="SARSA train")
    ax[0].plot(list(range(len(dq_tr))), dq_tr, label="Double Q-Learning train")
    ax[0].legend(loc="lower right")
    ax[1].plot(list(range(len(q_te))), q_te, label="Q-Learning test")
    ax[1].plot(list(range(len(sarsa_te))), sarsa_te, label="SARSA test")
    ax[1].plot(list(range(len(dq_te))), dq_te, label="Double Q-Learning test")
    ax[1].legend(loc="lower right")
    plt.savefig(f"{env_name}.png")
    plt.close()
    
def getPolicies(qQ, sQ, dqQ, cfg, world):
    mapqQ = np.argmax(qQ, axis=1)
    mapqQ = mapqQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    mapsQ = np.argmax(sQ, axis=1)
    mapsQ = mapsQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    mapdqQ = np.argmax(dqQ, axis=1)
    mapdqQ = mapdqQ.reshape(cfg["world_size"][0], cfg["world_size"][1]) # they are reversed
    
    actions_mapping = {
        0: '<-',
        1: 'v',
        2: '->',
        3: '^'
    }

    cmap = plt.get_cmap('tab10', len(actions_mapping))

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    for ax, matrix in zip(axs[:-1], [mapqQ, mapsQ, mapdqQ]):
        ax.imshow(matrix, cmap=cmap, interpolation='nearest')
    cbar = plt.colorbar(axs[2].imshow(mapqQ, cmap=cmap, interpolation='nearest'), ax=axs[-1], fraction=0.03, pad=0.04, ticks=np.arange(len(actions_mapping)))
    cbar.ax.set_yticklabels([actions_mapping[i] for i in range(len(actions_mapping))])

    axs[0].set_title('Q Learning')
    axs[1].set_title('Sarsa')
    axs[2].set_title('Double Q Learning')
    for ax in axs:
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')
    plt.tight_layout()
    plt.savefig(f"Policies_{world}.jpeg")
    plt.show()
    plt.close()
    
def main():

    cfg = {
        "num_actions": 4,
        "world_size": [7, 10],
        "world_type": "B"
    }
    env = GridWorld(cfg=cfg)

    random.seed(0)
    names = ["Q-Learning", "SARSA", "Double Q-Learning"]

    gamma = 0.9
    epsilon = 0.1
    alpha = 0.9
    num_steps = 15000
    eval_iter = 100
    for name in names:
        if name == "Q-Learning":
            q_tr, q_te, qlens, qQ = q_learning(env, gamma, epsilon, alpha, num_steps, eval_iter)
        elif name == "SARSA":
            sarsa_tr, sarsa_te, slens, sQ = sarsa(env, gamma, epsilon, alpha, num_steps, eval_iter)
        elif name == "Double Q-Learning":
            dq_tr, dq_te, dqlens, dqQ = double_q_learning(env, gamma, epsilon, alpha, num_steps, eval_iter)

    getPolicies(qQ, sQ, dqQ, cfg, cfg['world_type'])
    plot_for_env(f"GridWorld_{cfg['world_type']}", q_tr, q_te, sarsa_tr, sarsa_te, dq_tr, dq_te)
    plt.figure()
    plt.plot(list(range(len(qlens))), qlens, label="Q lengths")
    plt.plot(list(range(len(slens))), slens, label="SARSA lengths")
    plt.plot(list(range(len(dqlens))), dqlens, label="Double Q lengths")
    plt.legend()
    plt.savefig(f"ep_lens_{cfg['world_type']}.png")
    plt.close()

if __name__=="__main__":
    main()