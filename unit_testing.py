import unittest
import numpy as np
import os
from main import preceptron
from main import binary_classification
from main import abstract_training_class
from main import Adaline_classification
from main import SGDbinary_classification
import pickle


# test where genrated by AI .Grmini 3

class TestPerceptronArchitectures(unittest.TestCase):

    def setUp(self):
        """Common test data: A linearly separable set (AND gate logic)."""
        self.X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        self.y = np.array([0, 0, 0, 1])  # AND gate
        self.shape = 2

    # --- 1. Initialization & Structure Tests ---

    def test_binary_init(self):
        """Verifies binary_classification correctly links to perceptron and sets weights."""
        model = preceptron(shape=self.shape, training_class=binary_classification)
        self.assertIsInstance(model.training_class, binary_classification)
        self.assertEqual(len(model.vector), self.shape)
        self.assertEqual(model.bias, 0)

    def test_adaline_init(self):
        """Verifies Adaline-specific cost initialization."""
        model = preceptron(shape=self.shape, training_class=Adaline_classification)
        self.assertTrue(hasattr(model.training_class, 'costs'))
        self.assertEqual(len(model.training_class.costs), 0)

    # --- 2. Functional Method Tests ---

    def test_binary_fit_flow(self):
        """Checks if binary_classification completes a fit cycle and records errors."""
        model = preceptron(shape=self.shape, n_itterations=10, training_class=binary_classification)
        model.training_class.fit(self.X, self.y)
        self.assertEqual(len(model.errors), 10)
        self.assertIsInstance(model.errors[0], (int, np.integer))

    def test_sgd_fit_flow(self):
        """Checks if SGD version completes a fit cycle."""
        model = preceptron(shape=self.shape, n_itterations=5, training_class=SGDbinary_classification)
        model.training_class.fit(self.X, self.y)
        self.assertEqual(len(model.errors), 5)


    # --- 3. Mathematical Edge Cases ---

    def test_prediction_boundary(self):
        """Tests the activation function logic (Step function)."""
        model = preceptron(shape=self.shape, training_class=binary_classification)
        # Manually set weights to test boundary
        model.vector = np.array([1.0, 1.0])
        model.bias = -1.5
        # [1, 1] dot [1, 1] + (-1.5) = 0.5 -> Should be 1
        # [0, 1] dot [1, 1] + (-1.5) = -0.5 -> Should be 0
        self.assertEqual(model.training_class.predict([1, 1]), 1)
        self.assertEqual(model.training_class.predict([0, 1]), 0)

    def test_high_dimensionality(self):
        """Verifies the system doesn't break with 100+ features."""
        large_shape = 150
        X_large = np.random.rand(10, large_shape)
        y_large = np.random.randint(0, 2, 10)
        model = preceptron(shape=large_shape, training_class=binary_classification)
        try:
            model.training_class.fit(X_large, y_large)
        except Exception as e:
            self.fail(f"High dimensional data failed: {e}")

    # --- 4. Persistence Tests ---

    def test_pickle_save_load(self):
        """Checks if the save/load mechanism preserves the training_class state."""
        fname = "temp_model.pkl"
        model = preceptron(shape=self.shape, training_class=binary_classification)
        model.vector = np.array([1.23, 4.56])
        
        # Test saving
        try:
            model.save(fname)
            self.assertTrue(os.path.exists(fname))
            
            # Test loading (using fixed logic)
            with open(fname, 'rb') as f:
                loaded = pickle.load(f)
            
            np.testing.assert_array_almost_equal(model.vector, loaded.vector)
            self.assertEqual(type(model.training_class), type(loaded.training_class))
        finally:
            if os.path.exists(fname):
                os.remove(fname)

if __name__ == "__main__":
    unittest.main()