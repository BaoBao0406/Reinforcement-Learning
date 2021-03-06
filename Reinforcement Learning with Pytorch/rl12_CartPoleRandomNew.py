import gym
import time
import torch
import random

env = gym.make('CartPole-v1')

seed_value = 23
env.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value)

num_episodes = 1000
steps_total = []

import matplotlib.pyplot as plt

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0
    
    while True:
        step += 1
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        # print(new_state)
        # print(info)
        # env.render()
        
        if done:
            steps_total.append(step)
            print('Espisode finished after %i steps' % step)
            break

print("Average reward: %.2f" % (sum(steps_total)/num_episodes))
print("Average reward (last 100 episodes): %.2f" % (sum(steps_total[-100:])/100))

plt.figure(figsize=(12,5))
plt.title("Rewards")
plt.bar(torch.arange(len(steps_total)), steps_total, alpha=0.6, color='green')
plt.show()

env.close()
env.env.close()