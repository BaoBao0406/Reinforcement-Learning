import gym
import matplotlib.pyplot as plt
import torch

env = gym.make('Taxi-v2')

number_of_states = env.observation_space.n
number_of_actions = env.action.n

Q = torch.zeros([number_of_states, number_of_actions])

num_episodes = 1000
step_total = []
reward_total = []
egreedy_total = []

gamma = 0.9
learning_rate = 0.9
egreedy = 0.7
egreedy_final = 0.1
egreedy_decay = 0.99

for i in range(num_episodes):
    state = env.reset()
    step = 0
    score = 0
    
    while True:
        step += 1
        random_for_egreedy = torch.rand(1)[0]
        
        if random_for_egreedy > egreedy:
            random_value = Q[state] + torch.rand(1, number_of_actions) / 1000
            action = torch.max(random_value, 1)[1][0]
            action = action.item()
        
        else:
            action = env.action_space.sample()
        
        if egreedy > egreedy_final:
            egreedy += egreedy_decay
        
        new_state, rewards, done, info = env.step(action)
        Q[state, action] = (1 - learning_rate) * Q[state, action] \
                            + learning_rate * (rewards + gamma * torch.max(Q[new_state]))
        score += rewards
        state = new_state
        
        if done:
            step_total.append(step)
            reward_total.append(rewards)
            egreedy_total.append(egreedy)
