from sys import maxint
from simulation.planner import RoutePlanner
from simulation.environment import Agent, Environment


class QLearner(object):
    def __init__(self, actions, init_q=0, gamma=1.0, alpha=1.0):
        self._q = {}
        self._actions = actions
        self._INIT_Q = init_q
        self._gamma = gamma
        self._alpha = alpha

    def _create_q_state(self):
        return dict([(action, self._INIT_Q) for action in self._actions])

    def _get_best_action(self, state):
        if state not in self._q:
            self._q[state] = self._create_q_state()

        return self._q[state].keys()[self._q[state].values().index(max(self._q[state].values()))]

    def get_action(self, state):
        """
        Predict what the best action is and return it
        :param state: The state of the environment to base the prediction on
        :return: Best action to take
        """
        return self._get_best_action(state)

    def learn(self, state, action, state_new, reward):
        """
        Learn to make better predictions
        :param state: State of the environment
        :param action: Action used
        :param state_new: New state of the environment
        :param reward: Reward achieved from making the action
        """
        new_state_best_action = self._get_best_action(state_new)
        learn_value = reward + self._gamma * self._q[state_new][new_state_best_action]

        if self._q[state][action] == self._INIT_Q:
            self._q[state][action] = learn_value
        else:
            self._q[state][action] += self._alpha * (learn_value - self._q[state][action])


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)
        self.color = 'red'
        self.planner = RoutePlanner(self.env, self)
        self.learner = QLearner(Environment.valid_actions, init_q=maxint, gamma=0.0)

    @staticmethod
    def _get_state(next_waypoint, inputs):
        # Ignore traffic coming from the right side.  We don't need this information for making a decision.
        oncoming_traffic = inputs["oncoming"] if inputs["oncoming"] != "right" else None

        return next_waypoint, inputs["light"], oncoming_traffic

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        next_waypoint = self.planner.next_waypoint()
        state = self._get_state(next_waypoint, self.env.sense(self))

        # Predict best action
        action = self.learner.get_action(state)

        # Execute action and get reward
        reward = self.env.act(self, action)
        new_state = self._get_state(self.planner.next_waypoint(), self.env.sense(self))

        # Train Agent
        self.learner.learn(state, action, new_state,  reward)
