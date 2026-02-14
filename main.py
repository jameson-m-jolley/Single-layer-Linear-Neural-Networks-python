from cProfile import label
import sys
import time
import numpy as np
import pandas as pd
import unittest
import pickle
from abc import ABC, abstractmethod


#this is the decision function 

class abstract_training_class(ABC):

    def __init__(self,preceptron,shape = 0):
        self.preceptron = preceptron
        self.preceptron.vector = np.random.normal(loc=0 ,scale=1, size = shape)

    def net_output(self,x):
        return np.dot(x, self.preceptron.vector) + self.preceptron.bias
    
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

#this is the more absstract preceptron and we can use this one to make other ones
#this is a genral one that has the correct structure that we can then use an adapter
#paddern to use a abstract class to make the correct traning algo
class preceptron(object):
    def __init__(self,shape,filepath, lerning_rate = 0.1, n_itterations = 50, seed = 0,training_class = None,  ):
        self.lerning_rate = lerning_rate
        self.n_itterations = n_itterations
        self.seed = seed
        self.errors = []
        self.vector = [] # this is just empty for now
        self.bias = 0
        self.training_class = training_class(shape = shape,preceptron = self)
        self.filepath = filepath 
        if(self.filepath):
            try:
                self = self.load()
                self.lerning_rate = lerning_rate
                self.n_itterations = n_itterations
                self.training_class = training_class(shape = shape,preceptron = self)
                self.filepath = filepath 
                print(f"{filepath} loaded into obj at {self}")
            except:
                print(f"no file with path:{filepath}")

    def __str__(self):
        return f"""
__________preceptron________________      
filepath        :{self.filepath}
lerning_rate    :{self.lerning_rate}
seed            :{self.seed}
vector          :{self.vector}
bias            :{self.bias}
training_class  :{self.training_class}
____________________________________ 
                """

    def save(self):
        with open(self.filepath,'wb') as f:
            pickle.dump(self,f)
    
    def load(self):
        with open(self.filepath,'rb') as f:
            return pickle.load(f)
  

class binary_classification(abstract_training_class):

    def __init__(self,preceptron,shape = 0):
        super().__init__(preceptron,shape)

    def fit(self,data_points,labes):
        # the book and the slides have us make the vector but that sould be done at the time of the obj creation
        # i dont like loops so i will use maps if i can
        def update(features,prediction):
                update = self.preceptron.lerning_rate*(int(prediction) - self.predict(features))
                self.preceptron.vector += update * features # this is an array op i didn't know this was a thing
                self.preceptron.bias += update
                return int(update != 0) # we return the errors
        for i in range(self.preceptron.n_itterations):
            errors = sum(map(update,data_points,labes))
            self.preceptron.errors.append(errors)
        return self

    def activation(self,z):
        return 1 if z >=0 else 0

    
class Adaline_classification(abstract_training_class):

    def __init__(self,preceptron,shape = 0):
        super().__init__(preceptron,shape)
        self.costs = [] # this is what we are to minimize 
        
    def fit(self,data_points,labes):
        data_points = np.asarray(data_points).astype(float)
        labes = np.asarray(labes).astype(float)
        # the book and the slides have us make the vector but that sould be done at the time of the obj creation
        for i in range(self.preceptron.n_itterations):
            try:
                errors =(labes -  self.net_output(data_points)) # the disetances of the output from the actctual
                print(i)
                self.preceptron.vector += self.preceptron.lerning_rate*data_points.T.dot(errors) # this is an array op i didn't know this was a thing
                self.preceptron.bias += self.preceptron.lerning_rate* sum(errors)
                cost = sum(errors**2)/2
                self.costs.append(cost)
                if i % 10 == 0:
                    print(f"Iteration {i}: Cost {cost}")
            except Exception as e:
                print(f"somthing was wrong with one of the itterations {e}")
        return self
    
    def activation(self, z):
        return super().activation(z)


class SGDbinary_classification(abstract_training_class):

    def __init__(self,preceptron,shape = 0):
        super().__init__(preceptron,shape)
        
    def fit(self,data_points,labes):
        # the book and the slides have us make the vector but that sould be done at the time of the obj creation
        # i dont like loops so i will use maps if i can
        def update(features,prediction):
                update = self.preceptron.lerning_rate*(int(prediction) - self.predict(features))
                self.preceptron.vector += update * features # this is an array op i didn't know this was a thing
                self.preceptron.bias += update
                return int(update != 0) # we return the errors
        for i in range(self.preceptron.n_itterations):
            errors = sum(map(update,data_points,labes))
            self.preceptron.errors.append(errors)            
        return self
    
    def activation(self, z):
        return super().activation(z)



    
def main():
    print("demo of the perceptron")
    data = pd.read_csv("irisdata.csv")
    subset = data.iloc[:, 0:4].values.astype(float)
    ## we need to ajust the labes for the classifiers 
    # Extract the raw species column once
    raw_species = data.iloc[:, 4].values 
    setosa_labels = (raw_species == "Iris-setosa").astype(int)
    versicolor_labels = (raw_species == "Iris-versicolor").astype(int)
    virginica_labels = (raw_species == "Iris-virginica").astype(int)
    ## we can then make the preceptrons
    setosa = preceptron(shape=4,training_class=binary_classification,filepath="setosa.pkl")
    versicolor = preceptron(shape=4,training_class=binary_classification,n_itterations= 50,lerning_rate=0.0001 , filepath="versicolor.pkl")
    virginica = preceptron(shape=4,training_class=binary_classification,n_itterations= 50,lerning_rate=0.0001,filepath="virginica.pkl")
    print("starting training of the preceptrons this may take a while")
    ## we should be able to train this now
    print("traning setosa...")
    setosa.training_class.fit(subset,setosa_labels)
    print(f"done error rate = {setosa.errors[-1]/len(setosa_labels)*100}%")
    setosa.save()
    print("traning versicolor...")
    versicolor.training_class.fit(subset,versicolor_labels)
    versicolor.save()
    print(f"done error rate = {versicolor.errors[-1]/len(setosa_labels)*100}%")
    print("traning virginica...")
    virginica.training_class.fit(subset,virginica_labels)
    virginica.save()
    print(f"done error rate = {virginica.errors[-1]/len(setosa_labels)*100}%")

    print("______done with the binary_classification demo______")
    print("training with Adaline")
    setosa_labels = (raw_species == "Iris-setosa").astype(int)
    versicolor_labels = (raw_species == "Iris-versicolor").astype(int)
    virginica_labels = (raw_species == "Iris-virginica").astype(int)
    setosa = preceptron(shape=4,training_class=Adaline_classification,filepath="setosa_Adaline.pkl")
    setosa.training_class.fit(subset.copy(),setosa_labels)
    setosa.save()
    versicolor = preceptron(shape=4,training_class=Adaline_classification, filepath="versicolor_Adaline.pkl")
    versicolor.training_class.fit(subset.copy(),versicolor_labels)
    versicolor.save()
    virginica = preceptron(shape=4,training_class=Adaline_classification,filepath="virginica_Adaline.pkl")
    virginica.training_class.fit(subset.copy(),virginica_labels)
    virginica.save()
    



if __name__ == "__main__":
    main()
