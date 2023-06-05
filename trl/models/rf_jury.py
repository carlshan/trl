import pickle
import torch
import numpy as np
import sklearn

class RandomForestUncertainty:

  def __init__(self,pickle_filepath=None, device = 'cpu'):
    if not pickle_filepath:
      assert("You must provide a pickle filepath to load, such as './rf_n=500.pkl'")
    
    self.pickle_filepath = pickle_filepath

    # Load from file
    with open(pickle_filepath, "rb") as file:
        loaded_model = pickle.load(file)

    self.rf_model = loaded_model
    # self.rf_tree_predictions = np.array([])

    self.device=device

  def uncertainty(self, input, discourse_type='std'):
    """
    Given an input which is a torch.tensor return a tensor 
    containing a float that represents the uncertainty estimation across 
    all the Random Forest estimators.
    """

    # The torch tensor will first be converted to a numpy array
    # so that the estimators can all use it.
    data = input.detach().cpu().numpy()

    with torch.no_grad():
      rf_tree_predictions = []
      for tree in self.rf_model.estimators_:
        probas = tree.predict_proba(data)
        # Probas is shape [# data, 2] as there are 2 classes
        rf_tree_predictions.append(probas)
      
      # Convert list of numpy arrays to a single 3D PyTorch tensor
      # Tensor shape should be (500, # data, 2)
      predictions_tensor = torch.tensor(rf_tree_predictions, device=self.device)
      # Calculate variance along the 0th dimension (i.e., across the 500 decision trees)
      uncertainty = predictions_tensor.std(dim=0)
      uncertainty = uncertainty[:, 0]

      return uncertainty