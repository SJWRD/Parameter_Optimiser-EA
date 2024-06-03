# Imports
from deap import base, creator, tools, algorithms
import numpy as np
import random
import pickle


# =======================================================
# GPU not supported yet
# =======================================================

# from pycuda import autoinit
# from pycuda.compiler import SourceModule

# # CUDA kernel code for crossover
# cuda_code = """
# __global__ void crossover_kernel(float* ind1, float* ind2, unsigned int n) {
#     unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
#     if (idx < n) {
#         float alpha = 0.5;  // Blend alpha value
#         float temp = ind1[idx];
#         ind1[idx] = alpha * ind1[idx] + (1 - alpha) * ind2[idx];
#         ind2[idx] = alpha * ind2[idx] + (1 - alpha) * temp;
#     }
# }
# """

# # Compile CUDA kernel code
# cuda_module = SourceModule(cuda_code)
# crossover_kernel = cuda_module.get_function("crossover_kernel")

# # Define the CUDA-based crossover function
# def cxBlendGPU(ind1, ind2):
#     n = len(ind1)
#     threads_per_block = 256
#     blocks_per_grid = (n + threads_per_block - 1) // threads_per_block
#     crossover_kernel(ind1, ind2, np.uint32(n), block=(threads_per_block, 1, 1), grid=(blocks_per_grid, 1))

# # Register the CUDA-based crossover function in DEAP
# tools.cxBlendGPU = cxBlendGPU

# =======================================================


class ParameterOptimizer:
    def __init__(self, param_ranges, population_size=50, generations=100, cxpb=0.5, mutpb=0.2, use_gpu=False):
        self.param_ranges = param_ranges
        self.population_size = population_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.use_gpu = use_gpu
        self.toolbox = base.Toolbox()
        self.setup_deap()

    
    def setup_deap(self):
        # Define the individual and the population
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        # Register the individual attribute generators
        for param_name, low, high in self.param_ranges:
            self.toolbox.register(param_name, random.uniform, low, high)

        # # Register the structure initializers
        self.toolbox.register("individual", tools.initCycle, creator.Individual,
                              [getattr(self.toolbox, param_name) for param_name, _, _ in self.param_ranges], n=1)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        # Register the evaluation, crossover, mutation, and selection functions
        self.toolbox.register("evaluate", self.step)

        if self.use_gpu:
            print("\n=== GPU not supported yet ===\n")
            # # Use GPU operators if requested
            # self.toolbox.register("mate", tools.cxBlendGPU, alpha=0.5)
            # self.toolbox.register("mutate", tools.mutPolynomialBoundedGPU,
            #                       low=[low for _, low, _ in self.param_ranges],
            #                       up=[high for _, _, high in self.param_ranges],
            #                       eta=1.0, indpb=0.2)
        # else:
        # Use CPU operators by default
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutPolynomialBounded,
                                low=[low for _, low, _ in self.param_ranges],
                                up=[high for _, _, high in self.param_ranges],
                                eta=1.0, indpb=0.2)

        self.toolbox.register("select", tools.selTournament, tournsize=3)


    def step(self, individual):
        # Extract the parameters from the individual
        params = {param_name: val for (param_name, _, _), val in zip(self.param_ranges, individual)}
        print(params)

        print("\n=== Replace the 'step' method with your own evaluation logic ===\n")

        # Placeholder for function evaluation logic
        # Override this method with an actual implementation
        score = random.random()  # Dummy score for demonstration

        return score,


    def save_state(self, filename, population, generation):
        with open(filename, 'wb') as f:
            pickle.dump((population, generation), f)
        # print(f"State saved to {filename}")


    def load_state(self, filename):
        with open(filename, 'rb') as f:
            population, generation = pickle.load(f)
        print(f"State loaded from {filename}")
        return population, generation


    def optimize(self, resume=False, save=False, state_file='optimizer_state.pkl', feedback=False):
        # Enable/disable feedback
        self.feedback = feedback
        
        # Load or create population
        if resume:
            # Load population and current generation
            population, start_gen = self.load_state(state_file)
        else:
            # Create an initial population
            population = self.toolbox.population(n=self.population_size)
            start_gen = 0

        # Statistics to keep track of the progress
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Print the header
        if self.feedback:
            print(f"\n{'gen':<8}{'nevals':<8}{'avg':<12}{'std':<8}{'min':<12}{'max':<12}")

        # ======================================================= START optimization loop
        # Loop for every generation
        for gen in range(self.generations):

            # Run the genetic algorithm
            population, logbook = algorithms.eaSimple(population, self.toolbox, self.cxpb, self.mutpb,
                                                1, stats=stats, verbose=False)
            
            # Get generation         
            gen = start_gen + gen

            # Print feedback
            if feedback:
                # Get the record from the logbook
                if len(logbook) > 1:
                    record = logbook[1]
                else:
                    record = logbook[0]
                # Print feedback
                print(f"{gen+1:<8}{record['nevals']:<8}{record['avg']:<12.4f}{record['std']:<8.3f}{record['min']:<12.4f}{record['max']:<12.4f}")

            # Save the current optimizer state
            if save:
                self.save_state(state_file, population, gen + 1)

        # ======================================================= END optimization loop

        # Get the best individual
        best_ind = tools.selBest(population, 1)[0]
            
        # Print the best individual's parameters
        if self.feedback:
            print("\n===========================\nBest individual parameters:")
            for name, value in zip(self.param_ranges, best_ind):
                print(f"{name[0]}: {value}")
            print("===========================\n")

        # Return the best individual
        return best_ind
