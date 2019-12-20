# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 09:03:32 2019

@author: MUKHESH
"""

import gym
import numpy as np
import random
from IPython.display import clear_output
import time

#setting the enviornment
env=gym.make("FrozenLake-v0")

#setting the action and states spaces
action_space=env.action_space.n
state_space=env.observation_space.n
#creating the q table for taking the action 
q_table=np.zeros((state_space,action_space))
#setting the episodes and time_perepisode
num_episodes=10000
step_per_episode=100
#setting the learning rate and discounted rate
learning_rate=0.1
discounted_rate=0.99
#setting the epison greedy streagy variables
exploration_rate=1
min_exploration_rate=0.01
max_exploration_rate=1
exploration_decay_rate=0.001

rewards_per_episodes=[]
for episode in range(num_episodes):
    state=env.reset()
    reward_per_episode=0
    done=False
    for step in range(step_per_episode):
        r=np.random.random()
        if r>exploration_rate:
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
        new_state,reward,done,info=env.step(action)
        q_table[state,action]=(q_table[state,action]*(1-learning_rate))+(learning_rate*(reward+discounted_rate*(np.max(q_table[new_state,:]))))
        state=new_state
        reward_per_episode+=reward
        if done==True:
#            print(episode,step)    
            break
    exploration_rate=min_exploration_rate+(max_exploration_rate-min_exploration_rate)*np.exp(-exploration_decay_rate*episode)
    rewards_per_episodes.append(reward_per_episode)

rewards_per_thousand=np.split(np.array(rewards_per_episodes),num_episodes/1000)    
count=1000
for r in rewards_per_thousand:
    print(str(count)+" : "+str(sum(r/1000)))
    count+=1000
 
for episode in range(3):
    clear_output(wait=True)
    print("Starting***episode"+str(episode)+"***")
    state=env.reset()
    time.sleep(1)
    done=False
    for step in range(step_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.3)
        action=np.argmax(q_table[state,:])
        new_state,reward,done,info=env.step(action)
        if done==True:
            clear_output(wait=True)
            env.render()
            if reward==1:
                print("you are reached the goal!Hey Hurray")
                time.sleep(3)
            else:
                print("you are fallen into frozen hole!Oh Shit")
                time.sleep(3)
                clear_output(wait=True)
            break
        state=new_state
env.close()
            
            
               