import numpy as np
from perceptron import perceptron, abstract_training_class
from settings import settings
import unittest

class SGDbinary_classification(abstract_training_class):

    def __init__(self,perceptron,shape = 0):
        super().__init__(perceptron,shape)
        self.cost = []
        
    def fit(self,data_points,labels):
        self.data_setup()
        data_points = np.asarray(data_points).astype(float)
        labels = np.asarray(labels).astype(float)
        for i in range(self.perceptron.n_iterations):
            try:
                suf_data_points,shf_labels = self.shuffle(data_points, labels)
                cost = []
                for _points,_labels in zip(suf_data_points,shf_labels):
                    cost.append(self.update_w(_points,_labels))
                avg_cost =sum(cost)/len(cost)
                self.cost.append(avg_cost)
                if i % (self.perceptron.n_iterations/settings.reporting_rate_scale) == 0:
                    self.save_CSV_data(i,cost=avg_cost)
                    print(f"{self.perceptron.filepath} Iteration {i}: Cost {avg_cost}")
            except Exception as e:
                print(f"something was wrong with one of the iterations {e}")
            
        return self 
    
    def shuffle(self, data_points, labels):
        data_points = np.array(data_points)
        labels = np.array(labels)
        r = np.random.permutation(len(labels))
        return data_points[r], labels[r]

    def update_w(self,data_point, label):
        error =label - self.net_output(data_point)
        self.perceptron.vector += self.perceptron.learning_rate * (data_point.dot(error))
        self.perceptron.bias += self.perceptron.learning_rate * error
        return 0.5 * error**2
    
    def activation(self, z):
        return 1 if z >=0.5 else 0


class test(unittest.TestCase):

    def setUp(self):
        """Common test data: A linearly separable set (AND gate logic)."""
        self.m = perceptron(4,"testSGD.json",training_class=SGDbinary_classification)

    def test_shuffle(self):
        arr1 = [x for x in range(10)]
        arr2 = [x for x in range(10)]
        x,y = self.m.training_class.shuffle(arr1, arr2)
        for x,y in zip(x,y):
            self.assertEqual(x,y)

        is_same_as_original = np.array_equal(x, arr1)
        self.assertFalse(is_same_as_original, "The shuffle failed to change the order.")



        
    pass

if __name__ == "__main__":
    unittest.main()
