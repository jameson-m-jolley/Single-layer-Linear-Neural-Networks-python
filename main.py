from cProfile import label
import sys
import time
import numpy as np
import pandas as pd
import unittest
import pickle
import json
from abc import ABC, abstractmethod


#this is the decision function 

class abstract_training_class(ABC):

    def __init__(self,preceptron,shape = 0):
        self.preceptron = preceptron
        self.preceptron.vector = np.random.normal(loc=0 ,scale=1, size = shape)

    def net_output(self,x):
        weights = np.array(self.preceptron.vector, dtype=np.float64)
        bias = float(self.preceptron.bias)
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
            except:
                print(f"no file with path:{filepath}")

    def to_JSON(self):
        clone = self.__dict__.copy()
        clone['training_class'] = None
        clone['vector'] = self.vector.tolist()
        return json.dumps(clone)

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
            bites = self.to_JSON().encode('utf-8')
            f.write(bites)
    
    def load(self):
        '''loads only the vector and the bias we dont need anything else from the file'''
        with open(self.filepath,'rb') as f:
            json_data = json.load(f)
            self.vector = np.array(json_data["vector"],dtype=float)
            self.bias = float(json_data["bias"])

  

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
                cost = sum(errors**2) / 2.0
                if cost == float('nan') or cost == float('inf') :
                    self.preceptron.lerning_rate = sys.float_info.min
                    raise ValueError("cost should not be nan or infinity lowering the learning rate to the min")
                self.preceptron.vector += self.preceptron.lerning_rate*data_points.T.dot(errors) # this is an array op i didn't know this was a thing
                self.preceptron.bias += self.preceptron.lerning_rate* sum(errors)
                
                self.costs.append(cost)
                if i % 10 == 0:
                    print(f"Iteration {i}: Cost {cost}")
            except Exception as e:
                print(f"somthing was wrong with one of the itterations {e}")
        return self
    
    def activation(self, z):
        return 1 if z >=0 else 0


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
        return 1 if z >=0 else 0



    
def train_all():
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
    setosa = preceptron(shape=4,training_class=binary_classification,filepath="setosa.json")
    versicolor = preceptron(shape=4,training_class=binary_classification,n_itterations= 50,lerning_rate=0.0001 , filepath="versicolor.json")
    virginica = preceptron(shape=4,training_class=binary_classification,n_itterations= 50,lerning_rate=0.0001,filepath="virginica.json")
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
    setosa = preceptron(shape=4,training_class=Adaline_classification,filepath="setosa_Adaline.json",n_itterations= 2500000,lerning_rate=0.0000001)
    setosa.training_class.fit(subset.copy(),setosa_labels)
    setosa.save()
    versicolor = preceptron(shape=4,training_class=Adaline_classification, filepath="versicolor_Adaline.json",n_itterations= 2500000,lerning_rate=0.0000001)
    versicolor.training_class.fit(subset.copy(),versicolor_labels)
    versicolor.save()
    virginica = preceptron(shape=4,training_class=Adaline_classification,filepath="virginica_Adaline.json",lerning_rate=0.0000001,n_itterations= 2500000)
    virginica.training_class.fit(subset.copy(),virginica_labels)
    virginica.save()
    


modles = {
    "setosa": preceptron(shape=4,training_class=binary_classification,filepath="setosa.json"),
    "setosa_Adaline": preceptron(shape=4,training_class=Adaline_classification,filepath="setosa_Adaline.json"),
    "setosa_SGD": preceptron(shape=4,training_class=SGDbinary_classification,filepath="setosa_SGD.json"),
    "versicolor": preceptron(shape=4,training_class=binary_classification, filepath="versicolor.json"),
    "versicolor_Adaline": preceptron(shape=4,training_class=Adaline_classification, filepath="versicolor_Adaline.json"),
    "versicolor_SGD":  preceptron(shape=4,training_class=SGDbinary_classification, filepath="versicolor_SGD.json"),
    "virginica" : preceptron(shape=4,training_class=binary_classification,filepath="virginica.json"),
    "virginica_Adaline" : preceptron(shape=4,training_class=Adaline_classification,filepath="virginica_Adaline.json"),
    "virginica_SGD" : preceptron(shape=4,training_class=SGDbinary_classification,filepath="virginica_SGD.json")
}

def predict(vec, modl =None):
    if modl == None:
        setosa = [modles["setosa"],modles["setosa_Adaline"],modles["setosa_SGD"]]
        versicolor = [modles["versicolor"],modles["versicolor_Adaline"],modles["versicolor_SGD"]]
        virginica = [modles["virginica"],modles["virginica_Adaline"],modles["versicolor_SGD"]]
        print("starting prediction with all the difrent preceptrons")
        voter_func = lambda preceptrons: sum(map(lambda x: x.training_class.predict(vec),preceptrons))
        vote_setosa = voter_func(setosa)
        vote_versicolor = voter_func(versicolor)
        vote_virginica = voter_func(virginica)
        print(f"votes \nsetosa:{vote_setosa} \nversicolor{vote_versicolor} \nvirginica{vote_virginica}")
    else:
        print(f"the prediction from {modl} is {modles[modl].training_class.predict(vec)}")

def main():
    help = ''''
Welcome to the demo
expected usage
[python command] [filename] [args]
Arguments start with a dash and can take values as arguments
-p [size vector]: takes int as the size of the vector and a vector of values to predict
-t              : runs the training part of the program
-m [string]     : runs a particular modle 
    options
    <
    setosa
    ,setosa_Adaline
    ,setosa_SGD
    ,versicolor
    ,versicolor_Adaline
    ,versicolor_SGD
    ,virginica
    ,virginica_Adaline
    ,virginica_SGD
    >
    '''

    modl = None

    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "-t":
                train_all()
            elif arg == "-p":
                n = int(sys.argv[i+1])
                vec = np.array(sys.argv[i+2:i+2+n],dtype=float)
                print(f"reading in {n} vals as a vector ->>{vec}")
                i+=n
            elif arg == "-m":
                modl = str(sys.argv[i+1])
                i += 1
                pass
            i += 1
            
        predict(vec,modl)
    else:
        print(help)
    pass

if __name__ == "__main__":
    main()
