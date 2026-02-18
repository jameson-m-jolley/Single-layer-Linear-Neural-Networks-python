
# 1. Run the training for all models
echo "--- Starting Global Training ---"
python3 main.py -t

echo -e "\n--- Testing Iris Predictions (Setosa) ---"
# Predicting a Setosa-like flower (Sepal length, Sepal width, Petal length, Petal width)
python3 main.py -m setosa -p 4 5.1 3.5 1.4 0.2  -test
python3 main.py -m setosa_Adaline -p 4 5.1 3.5 1.4 0.2 -test
python3 main.py -m setosa_SGD -p 4 5.1 3.5 1.4 0.2 -test

python3 main.py -m versicolor -test
python3 main.py -m versicolor_Adaline -test
python3 main.py -m versicolor_SGD  -test

python3 main.py -m virginica -test
python3 main.py -m virginica_Adaline -test
python3 main.py -m virginica_SGD  -test


echo -e "\n--- Testing Dropout/Graduate Predictions ---"
# Providing a dummy 35-length vector for the Dropout model
# (Adjust these numbers to match a real standardized row from your data2.csv)
VEC_35="1 17 5 171 1 1 122.0 1 19 12 5 9 127.3 1 0 0 1 1 0 20 0 0 0 0 0 0.0 0 0 0 0 0 0.0 0 10.8 1.4 1.74 "
python3 main.py -m Dropout -p 35 $VEC_35 -test
python3 main.py -m Graduate -p 35 $VEC_35 -test
python3 main.py -m Dropout_Adaline -p 35 $VEC_35 -test
python3 main.py -m Graduate_Adaline -p 35 $VEC_35 -test
python3 main.py -m Dropout_SGD -p 35 $VEC_35 -test
python3 main.py -m Graduate_SGD -p 35 $VEC_35 -test
echo "was Dropout"



echo -e "\n--- Tests Completed ---"
