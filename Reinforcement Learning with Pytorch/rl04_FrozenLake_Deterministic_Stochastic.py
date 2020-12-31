import gym
import matplotlib.pyplot as plt
import time

env = gym.make('FrozenLake-v0')
# Set Slippery to False
# env = gym.make('FrozenLake-v0', is_slippery=False)

num_episodes = 1000
steps_total = []

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0
    
    while True:
        step += 1
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        time.sleep(0.4)
        env.render()
        print(new_state)
        print(info)
        
        if done:
            steps_total.append(step)
            print('Espisode finished after %i steps' % step)
            break


print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()