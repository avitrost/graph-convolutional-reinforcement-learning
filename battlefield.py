from pettingzoo.magent import combined_arms_v5
from numpy import sqrt, square
import numpy as np
from time import sleep
from buffer import ReplayBuffer

VIEW_RADIUS = 5
TEAM_COLORS = ['red', 'blue']
CLASSES = ['ranged', 'mele']

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
    agents = []
    for a in positions:
        if not a == agent:
            if distance(positions[agent], positions[a]) <= VIEW_RADIUS:
                agents.append(a)
    return agents

#TODO split up the dicts of id/name by team
#TODO define sizes of action spaces for ranged vs melee
def train(env, names_to_ids, ids_to_names, team_size, team1_class1_model, team1_class2_model, team2_class1_model, team2_class2_model):
    max_episodes = 1000
    epsilon = 0.9

    #TODO size of replay buffer
    #TODO different replay buffers for each team
    replay_buffer = ReplayBuffer() 
    observations = env.reset()

    positions = get_agent_positions(env)
    adj_matrix_1 = build_adjacency_matrix(names_to_ids, positions)
    adj_matrix_2 = build_adjacency_matrix(names_to_ids, positions)

    for step in range(max_episodes):
        env.render()
        
        if step > 100:
            epsilon -= 0.0004
            if epsilon < 0.1:
                epsilon = 0.1

        input_matrix_1 = np.zeros([team_size, 1521])
        input_matrix_2 = np.zeros([team_size, 1521])
        
        #TODO append class of agent to end of observation space
        for agent in env.agents:
            if TEAM_COLORS[0] in agent:
                input_matrix_1[names_to_ids[agent] % team_size] = np.reshape(observations[agent], [-1])
            elif TEAM_COLORS[1] in agent:
                input_matrix_2[names_to_ids[agent] % team_size] = np.reshape(observations[agent], [-1])
        

        q_1_1 = team1_class1_model(input_matrix_1, adj_matrix_1)
        q_1_2 = team1_class2_model(input_matrix_1, adj_matrix_1)
        q_2_1 = team2_class1_model(input_matrix_2, adj_matrix_2)
        q_2_2 = team2_class2_model(input_matrix_2, adj_matrix_2)

        actions = {}

        for agent in env.agents:
            if CLASSES[0] in agent:
                if np.random.rand() < epsilon:
                    action = np.random.randint(n_actions_1)
                else:
                    if TEAM_COLORS[0] in agent:
                        action = tf.math.argmax(q_1_1[names_to_ids[agent]])
                    elif TEAM_COLORS[1] in agent:
                        action = tf.math.argmax(q_2_1[names_to_ids[agent]])
            elif CLASSES[1] in agent:
                if np.random.rand() < epsilon:
                    action = np.random.randint(n_actions_2)
                else:
                    if TEAM_COLORS[0] in agent:
                        action = tf.math.argmax(q_1_2[names_to_ids[agent]])
                    elif TEAM_COLORS[1] in agent:
                        action = tf.math.argmax(q_2_2[names_to_ids[agent]])
            actions[agent] = action


        next_observations, rewards, dones, infos = env.step(actions)
        positions = get_agent_positions(env)
        next_adj_matrix_1 = build_adjacency_matrix(names_to_ids, positions)
        next_adj_matrix_2 = build_adjacency_matrix(names_to_ids, positions)
        
        replay_buffer.add(observations, actions, rewards, next_observations, adj_matrix_1, next_adj_matrix_1, infos)
        
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
    names_to_ids = {}
    ids_to_names = {}
    for i, name in enumerate(agent_names):
        names_to_ids[name] = i
        ids_to_names[i] = name
    return names_to_ids, ids_to_names

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

    names_to_ids, ids_to_names = get_agent_id_maps(agent_names)

    train(env, names_to_ids, ids_to_names, team_size, None, None, None, None)


if __name__ == '__main__':
    main()