from cProfile import label
import sys
import numpy as np
from settings import settings
from perceptron import abstract_training_class

class Adaline_classification(abstract_training_class):

    def __init__(self,perceptron,shape = 0):
        super().__init__(perceptron,shape)
        self.costs = [] # this is what we are to minimize 
        
    def fit(self,data_points,labels):
        self.data_setup()
        data_points = np.asarray(data_points).astype(float)
        labels = np.asarray(labels).astype(float)
        # the book and the slides have us make the vector but that sould be done at the time of the obj creation
        for i in range(self.perceptron.n_iterations):
            try:
                errors =(labels -  self.net_output(data_points)) # the disetances of the output from the actctual
                cost = sum(errors**2) / 2.0
                if cost == float('nan') or cost == float('inf') :
                    self.perceptron.learning_rate = sys.float_info.min
                    raise ValueError("cost should not be nan or infinity lowering the learning rate to the min")
                self.perceptron.vector += self.perceptron.learning_rate*data_points.T.dot(errors) # this is an array op i didn't know this was a thing
                self.perceptron.bias += self.perceptron.learning_rate* sum(errors)
                
                self.costs.append(cost)
                if i % (self.perceptron.n_iterations/settings.reporting_rate_scale) == 0:
                    print(f"{self.perceptron.filepath} Iteration {i}: Cost {cost}")
                    self.save_CSV_data(i,cost=cost)
            except Exception as e:
                print(f"somthing was wrong with one of the itterations {e}")
        return self
    
    
    def activation(self, z):
        return 1 if z >=0.5 else 0
