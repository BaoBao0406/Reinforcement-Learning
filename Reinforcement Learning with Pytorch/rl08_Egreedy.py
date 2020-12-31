import gym
import matplotlib.pyplot as plt
import torch
import time

env = gym.make('FrozenLake-v0', is_slippery=False)

number_of_states = env.observation_space.n
number_of_actions = env.action_space.n

Q = torch.zeros([number_of_states, number_of_actions])

steps_total = []
rewards_total = []

num_episodes = 1000
gamma = 0.9
egreedy = 0.1

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0
    
    while True:
        step += 1
        random_for_egreedy = torch.rand(1)[0]
        
        # If random value taken out from torch is bigger than egreedy, use Optimal reward action
        if random_for_egreedy > egreedy:    
            random_values = Q[state] + torch.rand(1,number_of_actions) / 1000        
            action = torch.max(random_values,1)[1][0]
            action = action.item()
        # If random value taken out from torch is smaller than egreedy, use random action
        else:
            action = env.action_space.sample()
                
        new_state, reward, done, info = env.step(action)        
        Q[state, action] = reward + gamma * torch.max(Q[new_state])        
        state = new_state
        
        if done:
            steps_total.append(step)
            rewards_total.append(reward)
            print('Espisode finished after %i steps' % step)
            break

print(Q)
print('Percent of episodes finished successfully: {0}'.format(sum(rewards_total)/num_episodes))
print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()