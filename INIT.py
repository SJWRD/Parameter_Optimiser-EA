import random
from FunctionOptimizer.FunctionOptimizer import FunctionOptimizer

# Settings
population_size = 30
generations = 100


# Example parameter ranges
param_config = [("param1", 0.1, 1.0),  
                ("param2", 1, 10),
                ("param3", 1, 100)]  # (param_name, min_value, max_value)


class CustomOptimizer(FunctionOptimizer):
    def step(self, individual):
        # Extract the parameters from the individual
        params = {param_name: random.uniform(min_val, max_val) for param_name, min_val, max_val in param_config}

        # Train the function and get the score
        score = random.random() # Replace with actual training and evaluation logic

        print(params["param2"])
        
        return score,


optimizer = CustomOptimizer(param_config, generations=generations, population_size=population_size)
best_parameters = optimizer.optimize()
print("Best parameters found: ", best_parameters)


# OR if you wat to save/load an optimiser state
best_parameters = optimizer.optimize(resume=True, save=True, state_file='optimizer_state.pkl')
