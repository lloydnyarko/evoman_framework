################################
# Team 97                      #
# Author: Lloyd Nyarko         #
# l.nyarko@student.vu.nl       #
################################

# imports framework
import sys, os
sys.path.insert(0, 'evoman') 
from environment import Environment
from demo_controller import player_controller

# import additional libs 
from deap import base, creator, tools, algorithms
import time
import numpy as np
import math

# headless -> not using visuals and thus making experiments faster
headless = True
if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

experiment_name = 'pilot'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(experiment_name=experiment_name, 
                  enemies=[2],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest")

ini = time.time()  # sets time marker

run_mode = "test"

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

dom_u = 1
dom_l = -1
npop = 100
gens = 5
mutation = 0.2
last_best = 0

toolbox = base.Toolbox()



def deap_functions():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax, lifepoints=1)

    toolbox.register("individual", tools.initRepeat, creator.Individual, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=npop)