from itertools import compress
import random
import time
import matplotlib.pyplot as plt
import numpy as np
from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_selection(items,knapsack_max_capacity,n_selection,population):
    fitness_values = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    total_fitness = sum(fitness_values)
    selection_p = [fitness(items,knapsack_max_capacity,individual)/total_fitness for individual in population]
    selected_individual = []
    for _ in range(n_selection):
        selected_individual.append(population[np.random.choice(len(population), p=selection_p)])
    return selected_individual
    
def crossover(parent_1,parent_2):
    crossover_point = random.randrange(len(parent_1))
    child_1 = parent_1[:crossover_point] + parent_2[crossover_point:]
    child_2 = parent_2[:crossover_point] + parent_1[crossover_point:]
    return child_1, child_2
 
def mutation(population,pMut):
    for individual in population:
        for s in range(len(individual)):
            checker = random.random()
            if(checker<pMut):
                individual[s] = not individual[s]
    
def select_best_individes(n_elite,population,items,knapsack_max_capacity):
    old_population = population
    old_population.sort(key = lambda ind:fitness(items, knapsack_max_capacity, ind), reverse=True)
    elite = old_population[:n_elite]
    return elite

items, knapsack_max_capacity = get_big()
print(items)

population_size = 100
generations = 200
n_selection = 20
n_elite = 1

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)

pMut = 0.1
for _ in range(generations):
    population_history.append(population)
    # TODO: implement genetic algorithm
    our_selection = roulette_selection(items,knapsack_max_capacity,n_selection,population)
    elite = select_best_individes(n_elite,population,items,knapsack_max_capacity)
    next_gen = []
    while (len(next_gen)) < len(our_selection):
        parent1, parent2 = random.sample(our_selection, 2)
        child_1, child_2 = crossover(parent1, parent2)
        next_gen.extend([child_1,child_2])
    mutation(next_gen,pMut)
    next_gen += elite
    
    refill_selection = roulette_selection(items,knapsack_max_capacity,population_size-len(next_gen),population)
    
    population = next_gen + refill_selection
    
    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 100
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
