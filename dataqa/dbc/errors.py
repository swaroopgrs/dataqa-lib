"""
DBC Service Integration Error Classes

Simple exception classes for DBC-specific errors.
"""


class DBCCallableError(Exception):
    """Exception raised when a DBC callable function fails."""
    pass


class DBCClientError(Exception):
    """Exception raised for client-level errors in DBC operations."""
    pass