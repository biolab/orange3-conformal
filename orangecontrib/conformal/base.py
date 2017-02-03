
class ConformalPredictor:
    """Base class for conformal predictors."""

    def __call__(self, example, eps):
        """Extending classes should implement this method to return predicted values
        for a given example and significance level."""
        raise NotImplementedError

    def predict(self, example, eps):
        """Extending classes should implement this method to return a prediction object.
        for a given example and significance level."""
        raise NotImplementedError
