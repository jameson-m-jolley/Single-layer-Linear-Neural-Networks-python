we are using venv
to run the project with venv

preform the fallowing commands

source venv/bin/activate
python main.py <args>

usage:
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