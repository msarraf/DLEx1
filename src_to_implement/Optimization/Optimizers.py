import numpy as np
class Sgd:
    def __init__(self,learning_rate):
        #if not learning_rate.is_float():
        if not isinstance(learning_rate, float):
            if not isinstance(learning_rate, int):
                raise TypeError("learning rate must be set to an float or int")
            self.learning_rate = learning_rate
    def calculate_update(self,weight_tensor, gradient_tensor):
        updated_weights = np.subtract(weight_tensor, (np.multiply(self.learning_rate, gradient_tensor)))
        return (updated_weights)








