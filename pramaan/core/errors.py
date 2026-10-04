class PramaanError(Exception):
    pass


class BudgetExceeded(PramaanError):
    pass


class SplitAccessError(PramaanError):
    pass


class AsOfViolation(PramaanError):
    pass


class ToolError(PramaanError):
    pass


class ConfigError(PramaanError):
    pass


class NeedsConfirmation(PramaanError):
    pass
