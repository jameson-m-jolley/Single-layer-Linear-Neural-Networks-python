from cProfile import label
import sys
import numpy as np
import pandas as pd
from perceptron import perceptron ,binary_classification
from Adaline import Adaline_classification
from SGD import SGDbinary_classification


def train_all():
    print("--- Starting Training Demo ---")
    data = pd.read_csv("irisdata.csv")
    subset = data.iloc[:, 0:4].values.astype(float)
    raw_species = data.iloc[:, 4].values 

    target_species = ["setosa", "versicolor", "virginica"]
    model_types = ["", "_Adaline", "_SGD"] # Suffixes matching your dict keys

    for model_type in model_types:
        print(f"\n--- Training Algorithm: {model_type.strip('_') if model_type else 'Perceptron'} ---")
        
        for species in target_species:
            dict_key = f"{species}{model_type}"
            target_name = f"Iris-{species}"
            labels = (raw_species == target_name).astype(int)    
            print(f"Training {dict_key} for {target_name}...")
            model = modles[dict_key]
            model.training_class.fit(subset.copy(), labels)
            model.save()
            if hasattr(model, 'errors') and len(model.errors) > 0:
                print(f"Done. Final error count: {model.errors[-1]}")
            elif hasattr(model.training_class, 'costs') and len(model.training_class.costs) > 0:
                print(f"Done. Final cost: {model.training_class.costs[-1]:.6f}")

    target_name = ["Dropout","Graduate"]
    data = pd.read_csv("data2.csv", sep=';')
    subset = standardize(data.iloc[:, 0:35].values.astype(float))
    raw_target_class = data.iloc[:, 36].values 
    model_types = ["", "_Adaline", "_SGD"] 
    
    for type in model_types:
        print(f"\n--- Training Algorithm: {type.strip('_') if type else 'Perceptron'} ---")
        for name in target_name:
            dict_key = f"{name}{type}"
            target_class = (raw_target_class.copy() == name).astype(int)
            print(f"Training {dict_key} for {name}...")
            model = modles[dict_key]
            model.training_class.fit(subset.copy(), target_class)
            model.save()
            if hasattr(model, 'errors') and len(model.errors) > 0:
                print(f"Done. Final error count: {model.errors[-1]}")
            elif hasattr(model.training_class, 'costs') and len(model.training_class.costs) > 0:
                print(f"Done. Final cost: {model.training_class.costs[-1]:.6f}")


    print("\n--- All models trained and saved successfully ---")


modles = {
    "setosa": perceptron(shape=4,training_class=binary_classification,filepath="setosa.json",learning_rate=0.00001),
    "setosa_Adaline": perceptron(shape=4,training_class=Adaline_classification,filepath="setosa_Adaline.json",learning_rate=0.00001),
    "setosa_SGD": perceptron(shape=4,training_class=SGDbinary_classification,filepath="setosa_SGD.json",learning_rate=0.00001),
    "versicolor": perceptron(shape=4,training_class=binary_classification, filepath="versicolor.json",learning_rate=0.00001),
    "versicolor_Adaline": perceptron(shape=4,training_class=Adaline_classification, filepath="versicolor_Adaline.json",learning_rate=0.00001),
    "versicolor_SGD":  perceptron(shape=4,training_class=SGDbinary_classification, filepath="versicolor_SGD.json",learning_rate=0.00001),
    "virginica" : perceptron(shape=4,training_class=binary_classification,filepath="virginica.json",learning_rate=0.00001),
    "virginica_Adaline" : perceptron(shape=4,training_class=Adaline_classification,filepath="virginica_Adaline.json",learning_rate=0.00001),
    "virginica_SGD" : perceptron(shape=4,training_class=SGDbinary_classification,filepath="virginica_SGD.json",learning_rate=0.00001 ),
    "Dropout": perceptron(shape=35,training_class=binary_classification,filepath="Dropout.json",learning_rate=0.00001, n_iterations=100),
    "Dropout_Adaline": perceptron(shape=35,training_class=Adaline_classification,filepath="Dropout_Adaline.json",learning_rate=0.00001, n_iterations=10000),
    "Dropout_SGD" :perceptron(shape=35,training_class=SGDbinary_classification,filepath="Dropout_SGD.json",learning_rate=0.00001, n_iterations=100),
    "Graduate": perceptron(shape=35,training_class=binary_classification,filepath="Graduate.json",learning_rate=0.00001, n_iterations=100),
    "Graduate_Adaline" :perceptron(shape=35,training_class=Adaline_classification,filepath="Graduate_Adaline.json",learning_rate=0.00001, n_iterations=10000),
    "Graduate_SGD": perceptron(shape=35,training_class=SGDbinary_classification,filepath="Graduate_SGD.json",learning_rate=0.00001, n_iterations=100)
}

def standardize(X):
    # Standardize each column: (x - mean) / std
    return (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def predict(vec, modl =None):
    if modl == None:
        setosa = [modles["setosa"],modles["setosa_Adaline"],modles["setosa_SGD"]]
        versicolor = [modles["versicolor"],modles["versicolor_Adaline"],modles["versicolor_SGD"]]
        virginica = [modles["virginica"],modles["virginica_Adaline"],modles["versicolor_SGD"]]
        print("starting prediction with all the difrent perceptrons")
        voter_func = lambda perceptrons: sum(map(lambda x: x.training_class.predict(vec),perceptrons))
        vote_setosa = voter_func(setosa)
        vote_versicolor = voter_func(versicolor)
        vote_virginica = voter_func(virginica)
        print(f"votes \nsetosa:{vote_setosa} \nversicolor{vote_versicolor} \nvirginica{vote_virginica}")
    else:
        if modl in ["Dropout","Dropout_Adaline","Dropout_SGD","Graduate","Graduate_Adaline","Graduate_SGD"]:
            vec = standardize(vec)    
            print("vector was standardized")
        print(f"the prediction from {modl} is {modles[modl].training_class.predict(vec)}")

def test_ds(modl_key):
    if modl_key not in modles:
        print(f"Model {modl_key} not found!")
        return

    # 1. Load the correct dataset and prepare features/labels
    if any(x in modl_key for x in ["Dropout", "Graduate"]):
        data = pd.read_csv("data2.csv", sep=';')
        subset = standardize(data.iloc[:, 0:35].values.astype(float))
        # The target names we are looking for are in the key (e.g., "Dropout")
        target_name = "Dropout" if "Dropout" in modl_key else "Graduate"
        raw_labels = data.iloc[:, 36].values 
    else:
        data = pd.read_csv("irisdata.csv")
        subset = data.iloc[:, 0:4].values.astype(float)
        # Match keys like 'setosa' to 'Iris-setosa'
        target_name = f"Iris-{modl_key.split('_')[0]}" 
        raw_labels = data.iloc[:, 4].values 

    y_true = (raw_labels == target_name).astype(int)
    model = modles[modl_key]
    y_pred = np.array([model.training_class.predict(row) for row in subset])
    accuracy = np.mean(y_true == y_pred) * 100
    pred_1s = np.sum(y_pred == 1)
    total = len(y_pred)
    print(f"\n--- Test Results for {modl_key} ---")
    print(f"Targeting: {target_name}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Predicted 'Positive' for {pred_1s} out of {total} samples ({(pred_1s/total)*100:.1f}%)")
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    print(f"True Positives: {tp} | False Positives: {fp}")
    print(f"True Negatives: {tn} | False Negatives: {fn}")

def main():
    help = ''''
Welcome to the demo
expected usage
[python command] [filename] [args]
Arguments start with a dash and can take values as arguments
-test           : dose a test
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
    ,Dropout
    ,Dropout_Adaline
    ,Dropout_SGD
    ,Graduate
    ,Graduate_Adaline
    ,Graduate_SGD
    >
    '''

    modl = None
    ds = False
    vec = None

    if len(sys.argv) > 1:
        i = 1
        while i < len(sys.argv):
            arg = sys.argv[i]
            if arg == "-t":
                train_all()
            elif arg == "-p":
                n = int(sys.argv[i+1])
                vec = np.array(sys.argv[i+2:i+2+n],dtype=float)
                i+=n
            elif arg == "-m":
                modl = str(sys.argv[i+1])
                i += 1
                pass
            elif arg == "-test":
                ds = True
                pass
            i += 1

        if ds:
            test_ds(modl)
            

        if modl != None:
            try:
                predict(vec,modl)
            except Exception as e:
                print(e)
                pass
    else:
        print(help)
    pass

if __name__ == "__main__":
    main()
