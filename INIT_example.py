# Imports
import random
from ParameterOptimizer.ParameterOptimizer import ParameterOptimizer

# Settings
population_size = 50
generations = 100

# Example parameters
# (param_name, min_value, max_value)
param_config = [("param1", 0.1, 1.0),  
                ("param2", 1, 10),
                ("example1", 1, 100)]  


# Create class and change 'step' to your own purpose
class CustomOptimizer(ParameterOptimizer):
    def step(self, individual):

        """
        Adjust the step method to your own purpose.
        The step method gets called for every individual.
        Here you can get the parameters from that individual and give it a score based on well it did.
        """

        # Extract the parameters from the individual
        params = {param_name: val for (param_name, _, _), val in zip(self.param_ranges, individual)}

        # Split parameters
        param1 = params["param1"]
        param2 = params["param2"]
        param3 = params["example1"]
        
        # Evaluate the parameters and give a score
        score = random.random() # Replace this with actual evaluation logic

        # Return score
        return score,

# Apply optimisation
optimizer = CustomOptimizer(param_config, generations=generations, population_size=population_size)
best_parameters = optimizer.optimize(feedback=True)

# OR if you want to save/load an optimiser state
# best_parameters = optimizer.optimize(resume=True, save=True, state_file='optimizer_state.pkl', feedback=True)
