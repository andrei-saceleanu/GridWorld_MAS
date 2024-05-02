import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import random

from tqdm import tqdm

def eps_greedy_policy(env, Q, state, epsilon):
    if random.random() < epsilon:
        actions = list(range(env.action_space.n)) 
        return random.choice(actions)
    else:
        return np.argmax(Q[state])

def eps_greedy_policy_dql(env, Q1, Q2, state, epsilon):
    if random.random() < epsilon:
        actions = list(range(env.action_space.n)) 
        return random.choice(actions)
    else:
        return np.argmax(Q1[state] + Q2[state])
    
def generate_best_policy(env, Q):
    policy = {state: None for state in range(env.observation_space.n)}

    for state in range(env.observation_space.n):
        policy[state] = np.argmax(Q[state])

    return policy

def simulate_policy(env, Q, iterations):
    epoch_rewards = []

    policy = generate_best_policy(env, Q)

    for i in range(iterations):
        state = env.reset()
        state = state[0]
        done = False

        total_reward = 0
        while not done:
            action = policy[state]
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            state = next_state

        epoch_rewards.append(total_reward)

    return np.mean(epoch_rewards)
    
def q_learning(env, gamma, epsilon, alpha, iterations, _evaluation_step = 100):
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    train_rewards = []
    crt_rewards = []
    test_rewards = []
    ep_lens = []
    lens = []

    for i in tqdm(range(iterations)):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        ep_len = 0
        while not done:
            action = eps_greedy_policy(env, Q, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]))
            state = next_state
            ep_len += 1


        if i % _evaluation_step == 0 and i != 0:
            test_rewards.append(simulate_policy(env, Q, iterations=_evaluation_step))
            train_rewards.append(np.mean(crt_rewards[-_evaluation_step:]))
            lens.append(np.mean(ep_lens[-_evaluation_step:]))
            
        crt_rewards.append(total_reward)
        ep_lens.append(ep_len)

    return train_rewards, test_rewards, lens, Q


def double_q_learning(env, gamma, epsilon, alpha, iterations, _evaluation_step = 100):
    
    Q1 = np.zeros((env.observation_space.n, env.action_space.n))
    Q2 = np.zeros((env.observation_space.n, env.action_space.n))

    train_rewards = []
    crt_rewards = []
    test_rewards = []
    ep_lens = []
    lens = []

    for i in tqdm(range(iterations)):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        ep_len = 0
        while not done:
            action = eps_greedy_policy_dql(env, Q1, Q2, state, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            if random.random() < 0.5:
                Q1[state, action] += alpha * (reward + gamma * Q1[next_state, np.argmax(Q2[next_state])] - Q1[state, action])
            else:
                Q2[state, action] += alpha * (reward + gamma * Q2[next_state, np.argmax(Q1[next_state])] - Q2[state, action])
            state = next_state
            ep_len += 1


        if i % _evaluation_step == 0 and i != 0:
            test_rewards.append(simulate_policy(env, (Q1 + Q2)/2, iterations=_evaluation_step))
            train_rewards.append(np.mean(crt_rewards[-_evaluation_step:]))
            lens.append(np.mean(ep_lens[-_evaluation_step:]))
            
        crt_rewards.append(total_reward)
        ep_lens.append(ep_len)

    return train_rewards, test_rewards, lens, (Q1 + Q2)/2

def sarsa(env, gamma, epsilon, alpha, iterations, eval_iter = 100):
    
    Q = np.zeros((env.observation_space.n, env.action_space.n))

    train_rewards = []
    crt_rewards = []
    test_rewards = []
    ep_lens = []
    lens = []

    for i in tqdm(range(iterations)):
        state = env.reset()
        state = state[0]
        done = False
        total_reward = 0

        action = eps_greedy_policy(env, Q, state, epsilon)
        ep_len = 0
        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_action = eps_greedy_policy(env, Q, next_state, epsilon)
            done = terminated or truncated

            total_reward += reward
            Q[state, action] = (1- alpha) * Q[state, action] + alpha * (reward + gamma * Q[next_state, next_action])
            state = next_state
            action = next_action
            ep_len += 1

        if i % eval_iter == 0 and i != 0:
            test_rewards.append(simulate_policy(env, Q, iterations=eval_iter))
            train_rewards.append(np.mean(crt_rewards[-eval_iter:]))
            lens.append(np.mean(ep_lens[-eval_iter:]))
        
        crt_rewards.append(total_reward)
        ep_lens.append(ep_len)

    return train_rewards, test_rewards, lens, Q