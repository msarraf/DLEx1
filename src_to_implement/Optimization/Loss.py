import numpy as np
class CrossEntropyLoss:
    def __init__(self):
        pass
    def forward(self,prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        self.label_tensor = label_tensor
        select_index = np.argwhere(self.label_tensor == 1)
        estimated_output = self.prediction_tensor[np.arange(len(self.prediction_tensor)), select_index[:, 1]]
        computed_Loss = -(np.log(estimated_output + np.finfo(float).eps))
        #computed_Loss = -(np.log(estimated_output + 4.406564584124654*e**(-324)))

        return np.sum(computed_Loss)

    def backward(self,label_tensor):
        self.label_tensor = label_tensor
        estimated_error = -1 * np.divide(self.label_tensor, self.prediction_tensor)
        return estimated_error










