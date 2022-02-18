class UndefinedTask(Exception):
    pass


class UndefinedRegularizer(Exception):
    pass


class UndefinedPreconditioner(Exception):
    pass


class ConfigurationError(Exception):
    pass


class MathematicalIncompatibility(ConfigurationError):
    pass


class AlgorithmStopped(Exception):
    pass
