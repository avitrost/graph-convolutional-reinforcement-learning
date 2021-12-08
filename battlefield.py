from pettingzoo.magent import combined_arms_v5
from numpy import sqrt, square
import numpy as np
from time import sleep
from buffer import ReplayBuffer
import tensorflow as tf
import sys

from model import AgentModel

VIEW_RADIUS = 5
TEAM_COLORS = ['red', 'blue']
CLASSES = ['mele', 'ranged']
CLASS1_ACTIONS = 9
CLASS2_ACTIONS = 25
OBSERVATION_SPACE_SIZE = 1521
MAX_EPISODES = 1000
CAPACITY = 1000
MAX_STEPS = 500
RENDER = False

HIDDEN_DIM = 128


class model:
    def __init__(self):
        self.test = 'test'
    def call(self):
        print(self.test)

Model = model()

def distance(agent1, agent2):
    return sqrt(square(agent2[0] - agent1[0]) + square(agent2[1] - agent1[1]))

def mask(obvservations, agent):
    for key in obvservations:
        raise Exception('test')

def get_agent_positions(env):
    positions = []
    for handle in env._all_handles:
        pos = env.env.get_pos(handle)
        positions.extend(pos)
    position_dict = dict(zip(env.agents, positions))
    return position_dict

def get_agents_in_radius(agent, positions):
    if not agent in positions:
        return []
    agents = []
    for a in positions:
        if not a == agent:
            if distance(positions[agent], positions[a]) <= VIEW_RADIUS:
                if (TEAM_COLORS[0] in agent and TEAM_COLORS[0] in a) or (TEAM_COLORS[1] in agent and TEAM_COLORS[1] in a):
                    agents.append(a)
    return agents

def build_observation_matrices(id_maps, observations, agents, team_size):
    input_matrix_1 = np.zeros([team_size, OBSERVATION_SPACE_SIZE])
    input_matrix_2 = np.zeros([team_size, OBSERVATION_SPACE_SIZE])
    #TODO append class of agent to end of observation space
    for agent in agents:
        if TEAM_COLORS[0] in agent:
            input_matrix_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]] = np.reshape(observations[agent], [-1])
        elif TEAM_COLORS[1] in agent:
            input_matrix_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]] = np.reshape(observations[agent], [-1])
    
    return input_matrix_1, input_matrix_2

def build_reward_matrices(id_maps, rewards, agents, team_size):
    reward_matrix_1 = np.zeros([team_size])
    reward_matrix_2 = np.zeros([team_size])

    for agent in agents:
        if TEAM_COLORS[0] in agent:
            reward_matrix_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]] = rewards[agent]
        elif TEAM_COLORS[1] in agent:
            reward_matrix_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]] = rewards[agent]
    
    return (reward_matrix_1, reward_matrix_2)

#TODO split up the dicts of id/name by team
#TODO define sizes of action spaces for ranged vs melee
def train(env, id_maps, team_size, team1_model, team2_model):
    
    epsilon = 0.9
    GAMMA = 0.99
    n_epoch = 5
    batch_size = 128

    O_1 = np.ones((batch_size, team_size, OBSERVATION_SPACE_SIZE))
    O_2 = np.ones((batch_size, team_size, OBSERVATION_SPACE_SIZE))
    Next_O_1 = np.ones((batch_size, team_size, OBSERVATION_SPACE_SIZE))
    Next_O_2 = np.ones((batch_size, team_size, OBSERVATION_SPACE_SIZE))
    Matrix_1 = np.ones((batch_size, team_size, team_size))
    Matrix_2 = np.ones((batch_size, team_size, team_size))
    Next_Matrix_1 = np.ones((batch_size, team_size, team_size))
    Next_Matrix_2 = np.ones((batch_size, team_size, team_size))

    score = 0
    
    for episode in range(MAX_EPISODES):
        print('episode:', episode)
        if RENDER:
            env.render()
        
        if episode > 100:
            epsilon -= 0.0004
            if epsilon < 0.1:
                epsilon = 0.1

        #TODO size of replay buffer
        #TODO different replay buffers for each team
        replay_buffer_1 = ReplayBuffer(CAPACITY) 
        replay_buffer_2 = ReplayBuffer(CAPACITY) 
        observations = env.reset()
        input_matrix_1, input_matrix_2 = build_observation_matrices(id_maps, observations, env.agents, team_size)
        
        positions = get_agent_positions(env)
        adj_matrix_1 = build_adjacency_matrix(id_maps[TEAM_COLORS[0]]['names_to_ids'], positions)
        adj_matrix_2 = build_adjacency_matrix(id_maps[TEAM_COLORS[1]]['names_to_ids'], positions)
        
        for step in range(MAX_STEPS):
            print('episode/step:', episode, step)
            print('{} agents', len(env.agents))
            q_1 = team1_model.model(input_matrix_1, adj_matrix_1)
            q_2 = team2_model.model(input_matrix_2, adj_matrix_2)

            actions = {}

            action_matrix_1 = np.zeros([team_size])
            action_matrix_2 = np.zeros([team_size])

            for agent in env.agents:
                if CLASSES[0] in agent and TEAM_COLORS[0] in agent:
                    if np.random.rand() < epsilon:
                        action = np.random.randint(CLASS1_ACTIONS)
                    else:
                        action = tf.math.argmax(q_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]][:CLASS1_ACTIONS])
                    action_matrix_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]] = action
                elif CLASSES[0] in agent and TEAM_COLORS[1] in agent:
                    if np.random.rand() < epsilon:
                        action = np.random.randint(CLASS1_ACTIONS)
                    else:
                        action = tf.math.argmax(q_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]][:CLASS1_ACTIONS])
                    action_matrix_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]] = action
                elif CLASSES[1] in agent and TEAM_COLORS[0] in agent:
                    if np.random.rand() < epsilon:
                        action = np.random.randint(CLASS2_ACTIONS)
                    else:
                        action = tf.math.argmax(q_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]])
                    action_matrix_1[id_maps[TEAM_COLORS[0]]['names_to_ids'][agent]] = action
                elif CLASSES[1] in agent and TEAM_COLORS[1] in agent:
                    if np.random.rand() < epsilon:
                        action = np.random.randint(CLASS2_ACTIONS)
                    else:
                        action = tf.math.argmax(q_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]])
                    action_matrix_2[id_maps[TEAM_COLORS[1]]['names_to_ids'][agent]] = action

                actions[agent] = action


            next_observations, rewards, dones, infos = env.step(actions)

            positions = get_agent_positions(env)
            next_adj_matrix_1 = build_adjacency_matrix(id_maps[TEAM_COLORS[0]]['names_to_ids'], positions)
            next_adj_matrix_2 = build_adjacency_matrix(id_maps[TEAM_COLORS[1]]['names_to_ids'], positions)
            
            next_input_matrix_1, next_input_matrix_2 = build_observation_matrices(id_maps, next_observations, env.agents, team_size)
            reward_matrix_1, reward_matrix_2 = build_reward_matrices(id_maps, rewards, env.agents, team_size)

            replay_buffer_1.add(input_matrix_1, action_matrix_1, reward_matrix_1, next_input_matrix_1, adj_matrix_1, next_adj_matrix_1, infos)
            replay_buffer_2.add(input_matrix_2, action_matrix_2, reward_matrix_2, next_input_matrix_2, adj_matrix_2, next_adj_matrix_2, infos)

            input_matrix_1 = next_input_matrix_1
            input_matrix_2 = next_input_matrix_2
            adj_matrix_1 = next_adj_matrix_1
            adj_matrix_2 = next_adj_matrix_2
            observations = next_observations

            score += np.sum(list(rewards.values())) # total score across both teams, should update later
        
        if episode % 20 == 0:
            print(score / 2000)
            score = 0
        if episode < 1:
            continue

        for e in range(n_epoch):
            print('epoch:', e)
            
            batch_1 = replay_buffer_1.getBatch(batch_size)
            batch_2 = replay_buffer_2.getBatch(batch_size)
            for j in range(batch_size):
                sample_1 = batch_1[j]
                sample_2 = batch_2[j]
                O_1[j] = sample_1[0]
                O_2[j] = sample_2[0]
                Next_O_1[j] = sample_1[3]
                Next_O_2[j] = sample_2[3]
                Matrix_1[j] = sample_1[4]
                Matrix_2[j] = sample_2[4]
                Next_Matrix_1[j] = sample_1[5]
                Next_Matrix_2[j] = sample_2[5]

            with tf.GradientTape() as tape:
                q_values_1 = team1_model.model(O_1, Matrix_1)
                q_values_2 = team2_model.model(O_2, Matrix_2)
                expected_q_values_1 = tf.identity(q_values_1)
                expected_q_values_2 = tf.identity(q_values_2)
                target_q_values_1 = tf.reduce_max(team1_model.target_model(Next_O_1, Next_Matrix_1), axis=2)[0]
                target_q_values_2 = tf.reduce_max(team2_model.target_model(Next_O_2, Next_Matrix_2), axis=2)[0]
                
                for j in range(batch_size):
                    sample_1 = batch_1[j]
                    sample_2 = batch_2[j]
                    for i in range(team_size):
                        expected_q_values_1[j][i][sample_1[1][i]] = sample_1[2][i] + (1-sample_1[6])*GAMMA*target_q_values_1[j][i]
                        expected_q_values_2[j][i][sample_2[1][i]] = sample_2[2][i] + (1-sample_2[6])*GAMMA*target_q_values_2[j][i]
                
                loss_1 = tf.reduce_mean(tf.math.square(q_values_1 - expected_q_values_1))
                loss_2 = tf.reduce_mean(tf.math.square(q_values_2 - expected_q_values_2))
                print('total loss: ' + str((loss_1 + loss_2).numpy()))
            gradients_1 = tape.gradient(loss_1, team1_model.model.trainable_variables)
            gradients_2 = tape.gradient(loss_2, team2_model.model.trainable_variables)
            team1_model.model.optimizer.apply_gradients(zip(gradients_1, team1_model.model.trainable_variables))
            team2_model.model.optimizer.apply_gradients(zip(gradients_2, team2_model.model.trainable_variables))

        if episode % 5 == 0:
            team1_model.update_target_model()
            team2_model.update_target_model()
            
    #TODO backprop model
def sort_agents(agent_names):
    team1_agents = {
        CLASSES[0]: [],
        CLASSES[1]: []
    }
    team2_agents = {
        CLASSES[0]: [],
        CLASSES[1]: []
    }
    for i in range(len(agent_names)):
        if TEAM_COLORS[0] in agent_names[i]:
            if CLASSES[0] in agent_names[i]:
                team1_agents[CLASSES[0]].append(agent_names[i])
            elif CLASSES[1] in agent_names[i]:
                team1_agents[CLASSES[1]].append(agent_names[i])
            else:
                print(agent_names[i])
                raise Exception('agent found with unknown class')
        elif TEAM_COLORS[1] in agent_names[i]:
            if CLASSES[0] in agent_names[i]:
                team2_agents[CLASSES[0]].append(agent_names[i])
            elif CLASSES[1] in agent_names[i]:
                team2_agents[CLASSES[1]].append(agent_names[i])
            else:
                print(agent_names[i])
                raise Exception('agent found with unknown class')
        else:
            raise Exception('agent found with unknown team')
    return (team1_agents, team2_agents)

def get_agent_id_maps(agent_names):
    maps = {
        TEAM_COLORS[0]: {
            "names_to_ids": {},
            "ids_to_names": {}
        }, 
        TEAM_COLORS[1]: {
            "names_to_ids": {},
            "ids_to_names": {}
        }
    }

    team1_counter = 0
    team2_counter = 0

    for name in agent_names:
        if TEAM_COLORS[0] in name :
            maps[TEAM_COLORS[0]]["names_to_ids"][name] = team1_counter
            maps[TEAM_COLORS[0]]["ids_to_names"][team1_counter] = name
            team1_counter = team1_counter + 1
        elif TEAM_COLORS[1] in name:
            maps[TEAM_COLORS[1]]["names_to_ids"][name] = team2_counter
            maps[TEAM_COLORS[1]]["ids_to_names"][team2_counter] = name
            team2_counter = team2_counter + 1
        else:
            raise Exception("agent not on known team")
    return maps

def build_adjacency_matrix(names_to_ids, positions):
    agent_names = names_to_ids.keys()
    matrix = np.zeros((len(agent_names), len(agent_names)))
    for agent in agent_names:
        agent_id = names_to_ids[agent]
        neighbors = get_agents_in_radius(agent, positions)
        for neighbor in neighbors:
            matrix[agent_id][names_to_ids[neighbor]] = 1
    return matrix

def main():
    # create our parallel environment so we get all observations at once
    env = combined_arms_v5.parallel_env(map_size=45, minimap_mode=False, 
    step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, 
    attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)

    team_size = env.num_agents // 2
    agent_names = env.possible_agents

    team1_agents, team2_agents = sort_agents(agent_names)

    id_maps = get_agent_id_maps(agent_names)

    team1_model = AgentModel(HIDDEN_DIM, CLASS2_ACTIONS)
    team2_model = AgentModel(HIDDEN_DIM, CLASS2_ACTIONS)

    train(env, id_maps, team_size, team1_model, team2_model)

    team1_model.save('team1_model')
    team2_model.save('team2_model')


if __name__ == '__main__':
    main()