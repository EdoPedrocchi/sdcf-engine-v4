"""Project-specific exceptions for configuration and data-contract failures.

The exceptions in this module identify conditions that make a data audit
invalid. They do not implement recovery, imputation, or vendor-specific
transformation behavior.
"""


class SdcfError(Exception):
    """Base class for errors raised by the SDCF package."""


class ConfigurationError(SdcfError):
    """Indicate that tracked configuration is missing, invalid, or ambiguous."""


class DataRootError(SdcfError):
    """Indicate that the configured external data root cannot be used safely."""


class DataContractError(SdcfError):
    """Indicate that a source file violates its declared structural contract."""
