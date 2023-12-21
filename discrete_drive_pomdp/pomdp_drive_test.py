import pomdp_py
import numpy as np
import sys



# human and robot are at an intersection (stop signs)
# they both start stopped (i.e., at position = 0)
# the game resets after one agent goes (i.e., at position = 1)


human_action = None


# return probability of action = "stop"
def human_policy(s, z):
    # # workaround for the case where s = (0, 0)
    if s[0] != 0 or s[1] != 0:
        return 0.5
    # z = 0 is a defensive human
    if z == 0:
        return 0.9
    # z = 1 is an aggressive human        
    else:
        return 0.1

# return updated z
def human_dynamics(s, z, phi):
    # phi = 0 is a human who never changes z
    if phi == 0:
        return z
    # phi = 1 is a human who changes
    else:
        # if a collision occurs
        if s[0] == 1 and s[1] == 1:
            return 0
        else:
            return z


class MyState(pomdp_py.State):
    def __init__(self, s, z, phi):
        self.s = s
        self.z = z
        self.phi = phi
    def __hash__(self):
        return hash((self.s, self.z, self.phi))
    def __eq__(self, other):
        if isinstance(other, MyState):
            return self.s == other.s and self.z == other.z and self.phi == other.phi
        return False
    def __repr__(self):
        return "(" + str(self.s[0]) + "," + str(self.s[1]) + "), " + str(self.z)  + ", " + str(self.phi)

class MyAction(pomdp_py.Action):
    def __init__(self, a):
        self.a = a
    def __hash__(self):
        return hash(self.a)
    def __eq__(self, other):
        if isinstance(other, MyAction):
            return self.a == other.a
        return False
    def __repr__(self):
        return "MyAction(%s)" % self.a

class MyObservation(pomdp_py.Observation):
    def __init__(self, s, a):
        self.s = s
        self.a = a
    def __hash__(self):
        return hash((self.s, self.a))
    def __eq__(self, other):
        if isinstance(other, MyObservation):
            return self.s == other.s and self.a == other.a
        return False
    def __repr__(self):
        return "(" + str(self.s[0]) + "," + str(self.s[1]) + "), " + self.a


# Observation model
class ObservationModel(pomdp_py.ObservationModel):

    def sample(self, next_state, action):
        s = next_state.s
        return MyObservation(s, human_action)


# Transition Model
class TransitionModel(pomdp_py.TransitionModel):

    def sample(self, state, action):
        s, z, phi = state.s, state.z, state.phi
        a = action.a
        p_stop = human_policy(s, z)
        z1 = human_dynamics(s, z, phi)

        # reset
        if s[0] > 0 or s[1] > 0:
            return MyState((0, 0), z1, phi)

        # move robot car
        s1 = [s[0], s[1]]
        if a == "go":
            s1[1] += 1

        # move human car
        global human_action
        if np.random.rand() < p_stop:
            human_action = "stop"
            return MyState(tuple(s1), z1, phi)
        else:
            s1[0] += 1
            human_action = "go"
            return MyState(tuple(s1), z1, phi)


# Reward Model
class RewardModel(pomdp_py.RewardModel):
    def _reward_func(self, state, action):
        s = state.s
        # no one goes
        if s[0] == 0 and s[1] == 0:
            return 0
        # human goes through first
        if s[0] == 1 and s[1] == 0:
            return -3
        # both go and we crash
        if s[0] == 1 and s[1] == 1:
            return -10
        # robot goes through first
        if s[0] == 0 and s[1] == 1:
            return +5

    def sample(self, state, action, next_state):
        return self._reward_func(state, action)


# Policy Model
class PolicyModel(pomdp_py.RolloutPolicy):
    """A simple policy model with uniform prior over a
       small, finite action space"""
    ACTIONS = [MyAction(s)
              for s in {"go", "stop"}]

    def sample(self, state):
        return np.random.choice(self.get_all_actions())

    def rollout(self, state, history=None):
        return self.sample(state)

    def get_all_actions(self, state=None, history=None):
        return PolicyModel.ACTIONS


# create problem instance
class MyProblem(pomdp_py.POMDP):

    def __init__(self, init_true_state, init_belief):
        agent = pomdp_py.Agent(init_belief,
                               PolicyModel(),
                               TransitionModel(),
                               ObservationModel(),
                               RewardModel())
        env = pomdp_py.Environment(init_true_state,
                                   TransitionModel(),
                                   RewardModel())
        super().__init__(agent, env, name="MyProblem")


def test_planner(my_problem, planner, nsteps):

    for i in range(nsteps):
            
        action = planner.plan(my_problem.agent)

        print("==== Step %d ====" % (i+1))
        print("True state:", my_problem.env.state)
        print("Belief:", my_problem.agent.cur_belief)
        print("Action:", action)

        # take action and transition
        reward = my_problem.env.state_transition(action, execute=True)
        print("Reward:", reward)

        # make observation
        real_observation = MyObservation(my_problem.env.state.s, human_action)
        print(">> Observation:",  real_observation)
        
        
        
        my_problem.agent.update_history(action, real_observation)
        # update belief
        planner.update(my_problem.agent, action, real_observation)
        if isinstance(planner, pomdp_py.POUCT):
            print("Num sims:", planner.last_num_sims)
            print("Plan time: %.5f" % planner.last_planning_time)


# human is initially aggressive, not sure if phi = 0 or 1
# if phi = 0 then we cannot change z
# if phi = 1 then we can make human defensive by crashing
init_true_state = MyState((0, 0), 1, 1)
init_belief = pomdp_py.Histogram({MyState((0, 0), 1, 0): 0.5,
                                  MyState((0, 0), 1, 1): 0.5})
horizon_H = 12
my_problem = MyProblem(init_true_state, init_belief)

print("** Testing POMCP **")
my_problem.agent.set_belief(pomdp_py.Particles.from_histogram(init_belief, num_particles=10000), prior=True)
pomcp = pomdp_py.POMCP(max_depth=horizon_H, discount_factor=1.0,
                       num_sims=100000, exploration_const=50,
                       rollout_policy=my_problem.agent.policy_model,
                       show_progress=True, pbar_update_interval=5000)
test_planner(my_problem, pomcp, nsteps=horizon_H)