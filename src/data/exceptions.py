"""Pipeline errors that must be shown clearly instead of inventing data."""


class MissingInputError(FileNotFoundError):
    """A required local file or folder is absent."""


class SchemaError(ValueError):
    """A discovered file does not contain the columns needed for a requested step."""


class DatasetSchemaError(SchemaError):
    """A dataset does not match the expected schema or required columns."""


class SplitLeakageError(ValueError):
    """A pair or patient occurs across multiple split partitions."""


class ConfigurationError(ValueError):
    """A requested pipeline option cannot be applied to the available columns."""

