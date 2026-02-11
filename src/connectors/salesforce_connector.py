"""Salesforce CRM connector using simple-salesforce."""

from __future__ import annotations

import pandas as pd
from simple_salesforce import Salesforce, SalesforceAuthenticationFailed, SalesforceExpiredSession

from src.connectors.base import CRMConnector, CRMAuthError, CRMConnectionError, CRMRateLimitError


class SalesforceConnector(CRMConnector):
    """Connect to Salesforce via username/password+token or OAuth access_token."""

    def __init__(self) -> None:
        self.client: Salesforce | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with Salesforce.

        Supports two modes controlled by ``credentials["mode"]``:
        * ``"token"`` – requires ``username``, ``password``, ``security_token``.
        * ``"oauth"`` – requires ``access_token``, ``instance_url``.
        """
        mode = credentials.get("mode", "token")
        try:
            if mode == "oauth":
                self.client = Salesforce(
                    instance_url=credentials["instance_url"],
                    session_id=credentials["access_token"],
                )
            else:
                self.client = Salesforce(
                    username=credentials["username"],
                    password=credentials["password"],
                    security_token=credentials["security_token"],
                )
            # Quick validation – describe() confirms auth works.
            self.client.describe()
            return True
        except SalesforceAuthenticationFailed as exc:
            raise CRMAuthError(f"Autenticacion fallida: {exc}") from exc
        except Exception as exc:
            raise CRMConnectionError(f"Error de conexion con Salesforce: {exc}") from exc

    # ------------------------------------------------------------------
    def test_connection(self) -> bool:
        if self.client is None:
            return False
        try:
            self.client.describe()
            return True
        except SalesforceExpiredSession:
            return False
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Object exploration
    # ------------------------------------------------------------------
    def list_objects(self) -> list[dict]:
        if self.client is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")
        desc = self.client.describe()
        objects = [
            {"name": obj["name"], "label": obj["label"]}
            for obj in desc["sobjects"]
            if obj.get("queryable")
        ]
        objects.sort(key=lambda o: o["label"])
        return objects

    # ------------------------------------------------------------------
    # Data fetch
    # ------------------------------------------------------------------
    def fetch_object_data(self, object_name: str, max_records: int = 10000) -> pd.DataFrame:
        if self.client is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")

        try:
            obj_desc = getattr(self.client, object_name).describe()
        except SalesforceExpiredSession as exc:
            raise CRMAuthError("La sesion de Salesforce expiro. Reconecta.") from exc
        except Exception as exc:
            raise CRMConnectionError(f"No se pudo describir el objeto '{object_name}': {exc}") from exc

        field_names = [f["name"] for f in obj_desc["fields"]]
        soql = f"SELECT {', '.join(field_names)} FROM {object_name} LIMIT {max_records}"

        try:
            result = self.client.query_all(soql)
        except Exception as exc:
            err_msg = str(exc)
            if "REQUEST_LIMIT_EXCEEDED" in err_msg:
                raise CRMRateLimitError("Limite de requests de Salesforce excedido. Espera unos minutos.") from exc
            raise CRMConnectionError(f"Error al consultar '{object_name}': {exc}") from exc

        records = result.get("records", [])
        # Remove Salesforce metadata key from each record
        for rec in records:
            rec.pop("attributes", None)

        if not records:
            return pd.DataFrame(columns=field_names)

        return pd.DataFrame(records)
