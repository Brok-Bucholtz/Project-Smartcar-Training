from learning_agent import LearningAgent
from simulation.environment import Environment
from simulation.simulator import Simulator


def run():
    # Set up environment and agent
    e = Environment()
    a = e.create_agent(LearningAgent)
    e.set_primary_agent(a, enforce_deadline=False)

    # Run simulation
    sim = Simulator(e, update_delay=1.0)
    sim.run(n_trials=10)


if __name__ == '__main__':
    run()
