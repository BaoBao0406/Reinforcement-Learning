import gym

env = gym.make('CartPole-v1')

num_episodes = 1000

# 1. First testing
# =============================================================================
# for i_episode in range(num_episodes):
#     state = env.reset()

#     for step in range(100):
#         action = env.action_space.sample()
#         new_state, reward, done, info = env.step(action)
#         env.render()
#         
#         if done:
#             break
# =============================================================================

# 2. Second testing and print out information
# =============================================================================
# for i_episode in range(num_episodes):
#     state = env.reset()
#     step = 0
#     
#     while True:
#         step += 1
#         action = env.action_space.sample()
#         new_state, reward, done, info = env.step(action)
#         
#         print(new_state)
#         print(info)
#         env.render()
#         
#         if done:
#             print('Espisode finished after %i steps' % step)
#             break
# =============================================================================

# 3. Third testing and create diagram for number of steps
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