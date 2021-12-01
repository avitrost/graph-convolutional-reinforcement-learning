from pettingzoo.magent import combined_arms_v5
from numpy import sqrt, square

class model:
    def __init__(self):
        self.test = 'test'
    def call(self):
        print(self.test)

Model = model()

def distance(agent1, agent2):
    return sqrt(square(agent2.x - agent1.x) + square(agent2.y - agent1.y))

# def mask(observations, agent):
#     for 

def mask(obvservations, agent):
    print(obvservations)
    raise Exception('test')

def train(env):
    max_cycles = 1000
    observations = env.reset()
    for step in range(max_cycles):
        actions = {}
        for agent in env.agents:
            obversvations_vector = mask(observations, agent)
            action = Model.call()
            

def main():
    # create our parallel environment so we get all observations at once
    # need minimap_mode enabled to get agent position in obvservation space!
    env = combined_arms_v5.parallel_env(map_size=45, minimap_mode=True, step_reward=-0.005, dead_penalty=-0.1, attack_penalty=-0.1, attack_opponent_reward=0.2, max_cycles=1000, extra_features=False)
    train(env)


if __name__ == '__main__':
    main()