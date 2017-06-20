import random
import math
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator

class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """

    def __init__(self, env, learning=False, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.default_q_value = 0.1
        self.trial_count = 0
        self.sucess_count = 0
        self.pre_state = None
        self.pre_action = None
        self.pre_reward = None


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)

        ###########
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        # Update additional class parameters as needed
        self.trial_count = self.trial_count + 1.0
        if self.env.success:
            self.sucess_count += 1
        self.pre_state = None
        self.pre_action = None
        self.pre_reward = None

        # If 'testing' is True, set epsilon and alpha to 0
        if testing:
            self.epsilon = 0
            self.alpha = 0
        else:
            #self.epsilon = self.epsilon - 0.05
            self.epsilon = math.pow(math.e, -0.02 * self.trial_count)
            self.alpha = self.alpha - 0.002

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the
            environment. The next waypoint, the intersection inputs, and the deadline
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ###########
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent
        #state = tuple(sorted(inputs.items()))
        state = (waypoint, inputs['light'], inputs['left'],inputs['right'], inputs['oncoming'])
        #state = (inputs['light'], inputs['oncoming'], waypoint)

        return state


    def get_maxQ(self, state):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ###########
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        return max(self.Q[state].values())



    def createQ(self, state):
        """ The createQ function is called when a state is generated by the agent. """

        ###########
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # If it is not, create a new dictionary for that state
        #   Then, for each action available, set the initial Q-value to 0.0
        if self.Q.get(state, None) == None:
            self.Q[state] = {}
            for  action in self.valid_actions:
                self.Q[state][action] = self.default_q_value

        return


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()

        ###########
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        random_action = random.choice(self.valid_actions)
        # 1, random, no learning
        if not self.learning:
           return random_action
        # 2, learning
        if random.random() < self.epsilon:
            print "++++ action by random for {} trial".format(self.trial_count)
            return random_action
        else:
            print "++++ action from max-q for {} trial".format(self.trial_count)
            state_actions = self.Q[state]
            max_q = self.get_maxQ(state)
            max_q_actions = [k for k,v in state_actions.items() if v == max_q]
            return random.choice(max_q_actions)


    def learn(self, state, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards
            when conducting learning. """

        ###########
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        if not self.learning:
            return

        if self.pre_state == None:
            return
        # Q-learing
        max_q = self.Q[state][action]
        #estimation = self.pre_reward + self.gamma * max_q
        estimation = self.pre_reward
        q_pre = self.Q[self.pre_state][self.pre_action]
        self.Q[self.pre_state][self.pre_action] = (1 - self.alpha) * q_pre +  self.alpha * estimation
        return


    def update(self):
        """ The update function is called when a time step is completed in the
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """
        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward) # Q-learn
        self.pre_state = state
        self.pre_action = action
        self.pre_reward = reward
        return


def run():
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment(verbose=True)

    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent, learning = True, epsilon = 0.6,alpha = 0.6)

    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent,enforce_deadline = True)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env, update_delay=0.01, log_metrics=True, display = False, optimized = True)

    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run(tolerance = 0.005, n_test=10)

    print "Total success rate: {:.2f}".format(agent.sucess_count / agent.trial_count * 100.)


if __name__ == '__main__':
    run()
