"""Abstract base class for CRM connectors."""

from abc import ABC, abstractmethod
import pandas as pd


class CRMAuthError(Exception):
    """Raised when CRM authentication fails."""


class CRMRateLimitError(Exception):
    """Raised when the CRM API rate limit is exceeded."""


class CRMConnectionError(Exception):
    """Raised when a connection to the CRM cannot be established."""


class CRMConnector(ABC):
    """Interface that every CRM connector must implement."""

    @abstractmethod
    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with the CRM using the provided credentials.

        Returns True on success; raises CRMAuthError on failure.
        """

    @abstractmethod
    def test_connection(self) -> bool:
        """Verify that the current session is still valid."""

    @abstractmethod
    def list_objects(self) -> list[dict]:
        """Return available CRM objects.

        Each element has the shape ``{"name": "Contact", "label": "Contacts"}``.
        The list is sorted alphabetically by *label*.
        """

    @abstractmethod
    def fetch_object_data(self, object_name: str, max_records: int = 10000) -> pd.DataFrame:
        """Fetch records for *object_name* and return them as a DataFrame.

        *max_records* caps the number of rows retrieved.
        """
