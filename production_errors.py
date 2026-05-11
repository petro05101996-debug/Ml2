class InputValidationError(ValueError):
    pass

class DataContractError(ValueError):
    pass

class ModelTrainingError(RuntimeError):
    pass

class ScenarioCalculationError(RuntimeError):
    pass

class DecisionAnalysisError(RuntimeError):
    pass

class ExportError(RuntimeError):
    pass
