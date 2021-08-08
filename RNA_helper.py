import RNA
import numpy as np
import pandas as pd

pairs_list = ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']
bases = ['G', 'U', 'A', 'C']
complements = {
    'G': ['C', 'U'], 
    'U': ['G', 'A'], 
    'A': ['U'], 
    'C': ['G']
}

def get_puzzle(df_eterna_100=None, eterna_id=None, idx=None, verbose=True, solution=1, return_name=False):
    if df_eterna_100 is None:
        df_eterna_100 = pd.read_csv('eterna100-puzzles.txt', delimiter='\t')
    if eterna_id is not None:
        row = df_eterna_100[df_eterna_100['Eterna ID'] ==  eterna_id].iloc[0]
    elif idx is not None:
        row = df_eterna_100.iloc[idx]
    else:
        print('eterna_id or idx must be set')
        return None, None
    secundary_structure = row['Secondary Structure']
    sequence = row[f'Sample Solution ({solution})']
    calc_secundary_structure, energy = RNA.fold(sequence)
    if verbose:
        print(row['Puzzle Name'])
        print(secundary_structure)
        print(sequence)
    if calc_secundary_structure != secundary_structure:
        energy_2 = RNA.eval_structure_simple(sequence, secundary_structure)
        print(idx, energy, energy_2)
        if energy == energy_2:
            print('yes')
        else:
            print('no')
        print()
        if verbose:
            print('*'*50, 'warning', '*'*50)
            print('It seems that RNA.fold(sequence) does not returns the same secundary structure. Sample Solution(1) folds to:')
            print(sequence)
            print('folds to:')
            print(calc_secundary_structure)
            print('annotated:')
            print(secundary_structure)
            print('*'*50, 'warning', '*'*50)
        return secundary_structure, None
    
    assert len(secundary_structure) == len(sequence)
    if return_name:
        return secundary_structure, sequence, row['Puzzle Name']
    return secundary_structure, sequence


def bracket_to_bonds(structure):
    bonds = [None]*len(structure)
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            bonds[i] = j
            bonds[j] = i
    return bonds



def get_initial_guess(secundary_structure, bonds_initial_type='random', verbose=True):
    sample_bases = bases
    if bonds_initial_type=='random':
        sample_pairs_list = pairs_list
        print('random pairs list')
    elif bonds_initial_type=='low_energy_bonds':
        sample_pairs_list = ['GU', 'UG']
        sample_bases = ['A']
#         print('low energy bonds pairs list')
    elif bonds_initial_type=='high_energy_bonds':
        sample_pairs_list = ['GC', 'CG']
        print('high energy bonds pairs list')
    elif bonds_initial_type=='all_As':
        print('All As')
        return 'A' * len(secundary_structure)
    else:
        print('default pairs list')
        sample_pairs_list = pairs_list
    bonds = bracket_to_bonds(secundary_structure)
    start_sequence = [None]*len(secundary_structure)
    for i, base in enumerate(start_sequence):
        if bonds[i] is None and base is None:
            start_sequence[i] = np.random.choice(sample_bases)
        elif base is None:
            sampled_pair = np.random.choice(sample_pairs_list)
            start_sequence[i] = sampled_pair[0]
            start_sequence[bonds[i]] = sampled_pair[1]
    start_sequence = ''.join(start_sequence)
    return start_sequence

def replace_char_towards_optimal(sequence, final_sequence, bonds={}):
    sequence_list = [c for c in sequence]
    sequence_array = np.array(sequence_list)
    final_sequence_array = np.array([c for c in final_sequence])
    indexes = list(np.where(sequence_array != final_sequence_array)[0])
    
#     print(indexes)
    idx = np.random.choice(indexes)
    replace_with = final_sequence[idx]
    sequence_list[idx] = replace_with
    if bonds[idx]:
        sequence_list[bonds[idx]] = final_sequence[bonds[idx]]
        
    return ''.join(sequence_list)

def move_to_optimal(start_sequence, objective_sequence, objective_secundary_structure):
    actual_sequence = start_sequence
    objective_bonds = bracket_to_bonds(objective_secundary_structure)
    energies = []
    free_energies = []
    sequences = []
    sec_struct_metric = []
    while actual_sequence != objective_sequence:
        energy = RNA.eval_structure_simple(actual_sequence, objective_secundary_structure)
        energies.append(energy)
        sequences.append(actual_sequence)
        
        actual_secundary_structure, free_energy = RNA.fold(actual_sequence)
        free_energies.append(free_energy)
        distance_metric = secundary_structures_metric(actual_secundary_structure, objective_secundary_structure)
        sec_struct_metric.append(distance_metric)
        if actual_secundary_structure == objective_secundary_structure:
            print('Found alternative sequence:')
            print(actual_sequence)
            print(free_energy)
            break
            
        print(f'{energy} - {distance_metric}\r', end='')
        actual_sequence = replace_char_towards_optimal(actual_sequence, objective_sequence, objective_bonds)
        

    actual_secundary_structure, free_energy = RNA.fold(actual_sequence)
    free_energies.append(free_energy)
    distance_metric = secundary_structures_metric(actual_secundary_structure, objective_secundary_structure)
    sec_struct_metric.append(distance_metric)
    print()
    print(distance_metric)
#     import ipdb; ipdb.set_trace()
    energies.append(RNA.eval_structure_simple(actual_sequence, objective_secundary_structure))
    sequences.append(actual_sequence)
    return energies, free_energies, sequences, sec_struct_metric


# METRICS

def bracket_to_bond_tuples(structure):
    bond_tuples = []
    opening = []
    for i,c in enumerate(structure):
        if c == '(':
            opening.append(i)
        elif c == ')':
            j = opening.pop()
            assert j < i
            bond_tuples.append((j, i))
    return set(bond_tuples), set()

def bracket_to_bonds_and_singles(structure):
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
    return set(bond_tuples), set(singles)

def secundary_structures_metric(current_structure, objective_structure):
    bond_tuples_current, singles_current = bracket_to_bonds_and_singles(current_structure)
    bond_tuples_objective, singles_objective = bracket_to_bonds_and_singles(objective_structure)
    bonds_intersection = len(bond_tuples_current.intersection(bond_tuples_objective))
    singles_intersection = len(singles_current.intersection(singles_objective))
    return (bonds_intersection + singles_intersection)/(len(bond_tuples_objective) + len(singles_objective))