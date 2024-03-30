import pandas as pd
import numpy as np
from evo import Environment
import random as rnd

# read the necessary files
tas = pd.read_csv('tas.csv', index_col = 0)
sections = pd.read_csv('sections.csv', index_col = 0)

pref = tas.iloc[:, 2:]

def overallocation(sol):
    """
    Calculate the overallocation penalty for a given solution.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.

    Returns:
    - int: Overallocation penalty for the solution.
    """
    diff = sol.sum(axis=1) - tas.max_assigned
    return sum(diff[diff > 0])


def undersupport(sol):
    """
    Calculate the undersupport penalty for a given solution.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.

    Returns:
    - int: Undersupport penalty for the solution.
    """
    diff = sections.min_ta - sol.sum(axis=0)
    return sum(diff[diff > 0])


def unpreferred(sol):
    """
    Calculate the penalty for assigning TAs to sections they prefer not to work in.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.

    Returns:
    - int: Penalty for assigning TAs to unpreferred sections.
    """
    return (sol * (pref == 'W').astype(int)).sum().sum()


def unwilling(sol):
    """
    Calculate the penalty for assigning TAs to sections they are unwilling to work in.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.

    Returns:
    - int: Penalty for assigning TAs to unwilling sections.
    """
    return ((sol * pref == 'U').astype(int).sum().sum())


def conflicts(sol):
    """
    Calculate the penalty for section assignments that result in conflicts in terms of daytime availability.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.

    Returns:
    - int: Penalty for conflicts in section assignments.
    """
    dt = sections['daytime'].values
    conflict = (dt[:, None] == dt).astype(int)
    ta_conflict = np.dot(sol, conflict) >= 2
    penalty = 0
    for x in ta_conflict:
        if True in x:
            penalty += 1
    return penalty

def swap_sections(solutions):
    """
    Swap two randomly selected sections in a solution.

    Args:
    - solutions (list): List of solutions.

    Returns:
    - numpy.ndarray: Solution after swapping two sections.
    """
    sol = solutions[0]
    num_sections = sol.shape[1]
    section1, section2 = rnd.sample(range(num_sections), 2)
    sol[:, [section1, section2]] = sol[:, [section2, section1]]
    return sol

def swap_sections(solutions):
    """
    Swap two randomly selected sections in a solution.

    Args:
    - solutions (list): List of solutions.

    Returns:
    - numpy.ndarray: Solution after swapping two sections.
    """
    sol = solutions[0]
    num_sections = sol.shape[1]
    section1, section2 = rnd.sample(range(num_sections), 2)
    sol[:, [section1, section2]] = sol[:, [section2, section1]]
    return sol


def random_assignment(solutions):
    """
    Generate a random assignment of TAs to sections.

    Args:
    - solutions (list): List of solutions.

    Returns:
    - numpy.ndarray: Randomly generated solution.
    """
    sol = np.random.choice([0, 1], size=solutions[0].shape)
    return sol


def combine_with_mutation(solutions):
    """
    Combine two solutions with mutation.

    Args:
    - solutions (list): List containing two solutions.

    Returns:
    - numpy.ndarray: Combined solution with mutation.
    """
    sol1, sol2 = solutions
    num_mutations = 2  
    mutation_probability = 0.05  
    sol = np.concatenate((sol1[:len(sol1) // 2], sol2[len(sol2) // 2:]), axis=0)
    for _ in range(num_mutations):
        if rnd.random() < mutation_probability:
            row_index = rnd.randrange(sol.shape[0])
            col_index = rnd.randrange(sol.shape[1])
            sol[row_index, col_index] = 1 - sol[row_index, col_index]
    return sol


def min_overallocate(solutions):
    """
    Eliminate overallocations of TAs in a solution by randomly reassigning sections.

    Args:
    - solutions (list): List of solutions.

    Returns:
    - numpy.ndarray: Solution after eliminating overallocations.
    """
    sol = solutions[0]
    diff = sol.sum(axis=1) - tas.max_assigned
    ta_ids = diff[diff > 0].index

    for num in ta_ids:
        (sol[num, :]) = np.random.choice([0,1], size=(17,), p=[.9, .1])
    return sol


def report(sol, eval):
    """
    Generate a report of the solution.

    Args:
    - sol (numpy.ndarray): 2D array representing the solution where rows correspond to TAs and columns correspond to sections.
    - eval (dict): Dictionary containing evaluation metrics.

    Returns:
    - tuple: Tuple containing TA dataframe, section dataframe, and evaluation metrics.
    """
    ta_dct = dict()
    sec_dct = dict()

    for i in range(len(sol)):
        ta_dct[tas.iloc[i, 0]] = (np.where(sol[i] == 1)[0])

    ta_df = pd.DataFrame({'ta': ta_dct.keys(), 'sections': ta_dct.values()})

    for j in range(len(sol.T)):
        assign = np.where(sol.T[j] == 1)[0]
        ta_lst = list()
        for ta_id in assign:
            ta_lst.append(tas.iloc[ta_id, 0])
        sec_dct[j] = ta_lst

    sec_df = pd.DataFrame({'section': sec_dct.keys(), 'tas': sec_dct.values()})
    sec_df = sec_df.set_index('section')

    eval_dict = {'Evaluation': eval}

    return ta_df, sec_df, eval_dict


def main():
    """
    Main function to run the evolutionary algorithm.
    """
    Ev = Environment()

    # Obj
    Ev.add_fitness_criteria('overallocation', overallocation)
    Ev.add_fitness_criteria('conflicts', conflicts)
    Ev.add_fitness_criteria('undersupport', undersupport)
    Ev.add_fitness_criteria('unwilling', unwilling)
    Ev.add_fitness_criteria('unpreferred', unpreferred)

    # agents
    Ev.add_agent("swap_sections", swap_sections, k=1)
    Ev.add_agent("random_assignment", random_assignment, k=1)
    Ev.add_agent("combine_with_mutation", combine_with_mutation, k=2)
    Ev.add_agent("min_overallocate", min_overallocate, k=1)

    row, col = pref.shape
    sol = np.random.choice([0, 1], size=(row, col), p=[.95, .05])
    Ev.add_solution(sol)

    Ev.evolve(100000000000000, 100, 10000)

    print(Ev)

    Ev.csv(groupname='tamiders')


if __name__ == '__main__':
    main()
