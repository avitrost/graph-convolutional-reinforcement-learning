from pettingzoo.magent import combined_arms_v5
from numpy import sqrt, square
from time import sleep

VIEW_RADIUS = 5

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

def train(env):
    max_cycles = 1000
    observations = env.reset()
    for step in range(max_cycles):
        env.render()

        agent_positions = get_agent_positions(env)
        focus_agent = env.agents[0]
        agents_in_radius = get_agents_in_radius(focus_agent, agent_positions)

        print("agents in view radius of %s:" % (focus_agent))
        print(agents_in_radius)

        actions = {}
        for agent in env.agents:
            obversvations_vector = mask(observations, agent)
            action = Model.call()
            

def main():
    # create our parallel environment so we get all observations at once
    env = combined_arms_v5.parallel_env(map_size=45, minimap_mode=False, 
    step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, 
    attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)
    train(env)


if __name__ == '__main__':
    main()