import gym

videosDir = './RLvideos/'
env = gym.make('CartPole-v1')
env = gym.wrappers.Monitor(env, videosDir)
num_episodes = 1000

import matplotlib.pyplot as plt

for i_episode in range(num_episodes):
    state = env.reset()
    step = 0
    steps_total = []
    
    while True:
        step += 1
        action = env.action_space.sample()
        new_state, reward, done, info = env.step(action)
        
        # print(new_state)
        # print(info)
        env.render()
        
        if done:
            steps_total.append(step)
            print('Espisode finished after %i steps' % step)
            break


print("Average number of steps: %.2f" % (sum(steps_total)/num_episodes))
plt.plot(steps_total)
plt.show()