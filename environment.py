import gym
import RNA
import numpy as np
from RNA_helper import bracket_to_bonds, secundary_structures_metric, get_initial_guess

import os


class RNAInvEnvironment(gym.Env):
    def __bracket_to_bonds_and_singles__(self, structure):
        bond_tuples = []
        opening = []
        singles = []
        for i,c in enumerate(structure):
            if c == '(':
                opening.append(i)
            elif c == ')':
                j = opening.pop()
                assert j < i
                bond_tuples.append((j, i))
            elif c == '.':
                singles.append(i)
            else:
                # No debería entrar
                assert False
        assert len(singles) +  2*len(bond_tuples) == len(structure)
        return bond_tuples, singles
    def __init__(self, objective_structure, policy='random', max_steps=100, tuple_obs_space=False, metric_type='energy', sequences_file='sequences_learned.txt'):
        self.sequences_file = sequences_file
        if not os.path.exists(sequences_file):
            f = open(self.sequences_file, 'w')
            f.close()
        self.pairs_list = np.array(['CG', 'GC', 'AU', 'UA', 'GU', 'UG'])
        self.bases_list = np.array(['G', 'U', 'A', 'C'])
        self.inv_bases_dict = {base: i for i, base in enumerate(self.bases_list)}
        self.N_pairs = len(self.pairs_list)
        self.N_bases = len(self.bases_list)
        self.objective_structure = objective_structure
        self.N = len(objective_structure)
        self.bonds = bracket_to_bonds(objective_structure)
        self.bond_tuples, self.singles = self.__bracket_to_bonds_and_singles__(objective_structure)
        self.pairs_count = len(self.bond_tuples)
        self.single_count = len(self.singles)
        self.action_space = gym.spaces.MultiDiscrete([self.N_pairs]*self.pairs_count + [self.N_bases]*self.single_count)
        self.tuple_obs_space = tuple_obs_space
        self.metric_type = metric_type
        
        if self.tuple_obs_space:
#             self.observation_space = gym.spaces.Tuple(
#                 tuple([gym.spaces.Discrete(self.N_bases) if i is None else gym.spaces.Discrete(self.N_pairs) for i in self.bonds])
#             )
#             self.observation_space = gym.spaces.MultiDiscrete(
#                 [self.N_bases if i is None else self.N_pairs for i in self.bonds]
#             )
            self.observation_space = gym.spaces.Box(low=np.array([0] * len(self.objective_structure)), high=np.array([self.N_bases if i is None else self.N_pairs for i in self.bonds]))
        else:
            self.observation_space = gym.spaces.Dict(
                {f'{k}': gym.spaces.Discrete(self.N_bases) if i is None else gym.spaces.Discrete(self.N_pairs) for k, i in enumerate(self.bonds)}
            )

        self.max_steps = max_steps

    def action_to_state(self, action):
        pair_predictions = self.pairs_list[action[:self.pairs_count]]
        single_predictions = self.bases_list[action[self.pairs_count:]]
        output = [None]*self.N
        for k, (i, j) in enumerate(self.bond_tuples):
            output[i] = pair_predictions[k][0]
            output[j] = pair_predictions[k][1]
        for k, i in enumerate(self.singles):
            output[i] = single_predictions[k]
        return ''.join(output)
    
    def action_to_state_tuple(self, action):
        pair_predictions = self.pairs_list[action[:self.pairs_count]]
        single_predictions = self.bases_list[action[self.pairs_count:]]
        
        return pair_predictions, single_predictions
    
    def sample_random_action(self):
        action = np.floor(self.action_space.sample()).astype(int)
        
        return self.action_to_state(action)
    
    def encode_state(self, state):
#         self.tuple_obs_space = False
        if self.tuple_obs_space:
            return [self.inv_bases_dict[base] for i, base in enumerate(state)]
        else:
            return {f'{i}': self.inv_bases_dict[base] for i, base in enumerate(state)}
    
    def reset(self):
#         self.state = get_initial_guess(self.objective_structure, 'low_energy_bonds')
        self.state = self.sample_random_action()
        self.steps = 0
        current_structure, energy = RNA.fold(self.state)
        self.objective_distance = secundary_structures_metric(current_structure, self.objective_structure)
        self.energy = RNA.eval_structure_simple(self.state, self.objective_structure)
        return self.encode_state(self.state)
    
    def step(self, action):
        self.steps = self.steps + 1
        self.state = self.action_to_state(action)
        current_structure, energy = RNA.fold(self.state)
        solved = (current_structure == self.objective_structure)
        done = solved or (self.steps >= self.max_steps)
        
        if done:
            if solved:
                print(self.state)
                f = open(self.sequences_file, 'a')
                f.write(self.state + '\n')
                
                f.close()

        new_objective_distance = secundary_structures_metric(current_structure, self.objective_structure)
        distance_reward = new_objective_distance - self.objective_distance
        self.objective_distance = new_objective_distance
        
        new_energy = RNA.eval_structure_simple(self.state, self.objective_structure)
        energy_reward = -(new_energy - self.energy)
        self.energy = new_energy
        
        if self.metric_type == 'energy':
            reward = energy_reward
        elif self.metric_type == 'distance':
            reward = distance_reward
        elif self.metric_type == 'total_distance':
            reward = new_objective_distance
        elif self.metric_type == 'combined':
            delta = 1 - new_objective_distance
            reward = new_objective_distance + 0.5 * delta * (energy_reward >= 0)
        elif self.metric_type == 'delta_energies':
            reward = (new_energy)
        else:
            print('Error, no metric found')
            assert False
        
        return self.encode_state(self.state), reward, done, {
            'free_energy': energy,
            'folding_struc': current_structure,
            'structure_distance': new_objective_distance,
            'energy_to_objective': new_energy,
            'energy_reward': energy_reward,
            'distance_reward': distance_reward,
            'sequence': self.state,
            'solved': solved
        }
    
    
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from boardgame2 import ReversiEnv

def make_vec_env(
    env_class,
    n_envs = 1,
    seed = None,
    start_index: int = 0,
    monitor_dir = None,
    wrapper_class = None,
    env_kwargs = None,
    vec_env_cls = None,
    vec_env_kwargs = None,
    monitor_kwargs = None,
    wrapper_kwargs = None,
):
   
    env_kwargs = {} if env_kwargs is None else env_kwargs
    vec_env_kwargs = {} if vec_env_kwargs is None else vec_env_kwargs
    monitor_kwargs = {} if monitor_kwargs is None else monitor_kwargs
    wrapper_kwargs = {} if wrapper_kwargs is None else wrapper_kwargs

    def make_env(rank):
        def _init():
            env = env_class(**env_kwargs)
               
            if seed is not None:
                env.seed(seed + rank)
                env.action_space.seed(seed + rank)
            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)
            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    return vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)

