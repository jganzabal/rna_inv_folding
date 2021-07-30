import RNA
import numpy as np

pairs_list = ['CG', 'GC', 'AU', 'UA', 'GU', 'UG']
bases = ['G', 'U', 'A', 'C']

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

def get_initial_guess(secundary_structure):
    bonds = bracket_to_bonds(secundary_structure)
    start_sequence = [None]*len(secundary_structure)
    for i, base in enumerate(start_sequence):
        if bonds[i] is None and base is None:
            start_sequence[i] = np.random.choice(bases)
        elif base is None:
            sampled_pair = np.random.choice(pairs_list)
            start_sequence[i] = sampled_pair[0]
            start_sequence[bonds[i]] = sampled_pair[1]
    start_sequence = ''.join(start_sequence)
    return start_sequence

def replace_char_towards_optimal(sequence, final_sequence):
    sequence_list = [c for c in sequence]
    idx = np.random.randint(len(final_sequence))
    sequence_list[idx] = final_sequence[idx]
    return ''.join(sequence_list)

def move_to_optimal(start_sequence, final_sequence, secundary_structure, calc_free_energy=False):
    sequence = start_sequence
    energies = []
    free_energies = []
    sequences = []
    while sequence != final_sequence:
        energy = RNA.eval_structure_simple(sequence, secundary_structure)
        energies.append(energy)
        sequences.append(sequence)
        if calc_free_energy:
            _, free_energy = RNA.fold(sequence)
            free_energies.append(free_energy)
        sequence = replace_char_towards_optimal(sequence, final_sequence)
        print(f'{energy}\r', end='')

    if calc_free_energy:
        _, free_energy = RNA.fold(sequence)
        free_energies.append(free_energy)
    energies.append(RNA.eval_structure_simple(sequence, secundary_structure))
    sequences.append(sequence)
    return energies, free_energies, sequences