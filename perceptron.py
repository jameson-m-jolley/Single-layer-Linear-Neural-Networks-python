#this is the decision function 
from cProfile import label
import numpy as np
import json
from abc import ABC, abstractmethod
from settings import settings


#this is just to save all the json files to one place 
models_path = "models/"
data_path = "data/"

class abstract_training_class(ABC):

    def __init__(self,perceptron,shape = 0):
        self.perceptron = perceptron
        self.perceptron.vector = np.random.normal(loc=0 ,scale=1, size = shape)

    def data_setup(self):
        with open(data_path+str(self)+".csv",'wb') as f:
            # epoch,errors,cost
            f.write("epoch,errors,cost\n".encode())
            #print("epoch,errors,cost", file=f)

    def save_CSV_data(self,epoch,errors = -1,cost = -1):
        with open(data_path+str(self)+".csv",'a') as f:
            # epoch,iteration,errors,cost
            print(f"{epoch},{errors},{cost}",file=f)


    def net_output(self,x):
        weights = np.array(self.perceptron.vector, dtype=np.float64)
        bias = float(self.perceptron.bias)
        if type(x) == str:
            print(f"type of x is a string that string is[{x}]")
        input_x = np.array(x, dtype=np.float64)
        return np.dot(input_x, weights) + bias
    
    def predict(self,input_vec = []):
        return self.activation(self.net_output(input_vec))
   
    @abstractmethod
    def fit(self):
        raise Exception("you need to impliment the fit function") 
        pass
  
    @abstractmethod
    def activation(self,z):
        raise Exception("you need to impliment the activation function") 
        pass

#this is the more absstract perceptron and we can use this one to make other ones
#this is a genral one that has the correct structure that we can then use an adapter
#paddern to use a abstract class to make the correct traning algo
class perceptron(object):
    def __init__(self,shape,filepath, learning_rate = 0.1, n_iterations = 50, seed = 0,training_class = None,  ):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.seed = seed
        self.errors = []
        self.vector = [] # this is just empty for now
        self.bias = 0
        self.training_class = training_class(shape = shape,perceptron = self)
        self.filepath = filepath 
        if(self.filepath):
            try:
                self = self.load()
            except:
                print(f"no file with path:{filepath}")

    def to_JSON(self):
        clone = self.__dict__.copy()
        clone['training_class'] = None
        clone['vector'] = self.vector.tolist()
        return json.dumps(clone)

    def __str__(self):
        return f"""
__________perceptron________________      
filepath        :{self.filepath}
learning_rate   :{self.learning_rate}
seed            :{self.seed}
vector          :{self.vector}
bias            :{self.bias}
training_class  :{self.training_class}
____________________________________ 
                """

    def save(self):
        with open(models_path+self.filepath,'wb') as f:
            bites = self.to_JSON().encode('utf-8')
            f.write(bites)
    
    def load(self):
        '''loads only the vector and the bias we dont need anything else from the file'''
        with open(models_path+self.filepath,'rb') as f:
            json_data = json.load(f)
            self.vector = np.array(json_data["vector"],dtype=float)
            self.bias = float(json_data["bias"])
 

class binary_classification(abstract_training_class):

    def __init__(self,perceptron,shape = 0):
        super().__init__(perceptron,shape)

    def fit(self,data_points,labels):
        self.data_setup()
        # the book and the slides have us make the vector but that sould be done at the time of the obj creation
        # i dont like loops so i will use maps if i can
        def update(features,prediction):
                update = self.perceptron.learning_rate*(int(prediction) - self.predict(features))
                self.perceptron.vector += update * features # this is an array op i didn't know this was a thing
                self.perceptron.bias += update
                return int(update != 0) # we return the errors
        for i in range(self.perceptron.n_iterations):
            errors = sum(map(update,data_points,labels))
            self.perceptron.errors.append(errors)
            if i % (self.perceptron.n_iterations/settings.reporting_rate_scale) == 0:
                print(f"{self.perceptron.filepath} Iteration {i}: Total errors {errors}")
                self.save_CSV_data(i,errors=errors)
        return self

    def activation(self,z):
        return 1 if z >=0 else 0