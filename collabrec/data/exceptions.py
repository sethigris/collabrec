# This file defines all custom errors that can occur in CollabRec.
# Each error has a clear name to indicate what went wrong.

class CollabRecError(Exception):
    """This is the base class for all errors in this project. Catching this catches any CollabRec problem."""

class DataError(CollabRecError):
    """Raised when there is a problem with the data file – for example, the file is missing, empty, or has the wrong columns."""

class ModelError(CollabRecError):
    """Raised when the recommendation model is used incorrectly, such as trying to predict before training."""

class UserNotFoundError(ModelError):
    """Raised when a requested user does not exist in the training data."""

class ItemNotFoundError(ModelError):
    """Raised when a requested item does not exist in the training data."""

class InsufficientDataError(ModelError):
    """Raised when there is not enough information to make recommendations – for example, the user has already rated every item."""