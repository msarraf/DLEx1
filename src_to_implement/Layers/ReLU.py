import numpy as np
class ReLU:
    def __init__(self):
        self.trainable = False


    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0,self.input_tensor)
    def backward(self,error_tensor):
        self.error_tensor = error_tensor
        self.error_tensor[self.input_tensor <= 0] = 0
        return self.error_tensor


