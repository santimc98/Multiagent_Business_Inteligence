"""HubSpot CRM connector using hubspot-api-client."""

from __future__ import annotations

import pandas as pd
from hubspot import HubSpot
from hubspot.crm.contacts import ApiException as ContactsApiException

from src.connectors.base import CRMConnector, CRMAuthError, CRMConnectionError, CRMRateLimitError


class HubSpotConnector(CRMConnector):
    """Connect to HubSpot via Private App token or OAuth access_token."""

    def __init__(self) -> None:
        self.client: HubSpot | None = None

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    def authenticate(self, credentials: dict) -> bool:
        """Authenticate with HubSpot.

        Requires ``access_token`` (Private App token or OAuth token).
        """
        token = credentials.get("access_token", "")
        if not token:
            raise CRMAuthError("Se requiere un access_token.")

        try:
            self.client = HubSpot(access_token=token)
            # Validate by fetching one contact
            self.client.crm.contacts.basic_api.get_page(limit=1)
            return True
        except ContactsApiException as exc:
            if exc.status == 401:
                raise CRMAuthError("Token invalido o expirado.") from exc
            raise CRMConnectionError(f"Error de conexion con HubSpot: {exc}") from exc
        except Exception as exc:
            raise CRMConnectionError(f"Error de conexion con HubSpot: {exc}") from exc

    # ------------------------------------------------------------------
    def test_connection(self) -> bool:
        if self.client is None:
            return False
        try:
            self.client.crm.contacts.basic_api.get_page(limit=1)
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Object exploration
    # ------------------------------------------------------------------
    def list_objects(self) -> list[dict]:
        if self.client is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")
        try:
            schemas = self.client.crm.schemas.core_api.get_all()
            objects = [
                {"name": s.name, "label": s.labels.get("plural", s.name) if isinstance(s.labels, dict) else s.name}
                for s in schemas.results
            ]
        except Exception:
            # Fallback: return well-known standard objects
            objects = []

        # Always include standard objects that may not appear in schemas
        standard = [
            {"name": "contacts", "label": "Contacts"},
            {"name": "companies", "label": "Companies"},
            {"name": "deals", "label": "Deals"},
            {"name": "tickets", "label": "Tickets"},
        ]
        existing_names = {o["name"] for o in objects}
        for std in standard:
            if std["name"] not in existing_names:
                objects.append(std)

        objects.sort(key=lambda o: o["label"])
        return objects

    # ------------------------------------------------------------------
    # Data fetch with cursor-based pagination
    # ------------------------------------------------------------------
    def fetch_object_data(self, object_name: str, max_records: int = 10000) -> pd.DataFrame:
        if self.client is None:
            raise CRMConnectionError("No autenticado. Llama a authenticate() primero.")

        # Discover properties for the object
        try:
            props_response = self.client.crm.properties.core_api.get_all(object_name)
            all_properties = [p.name for p in props_response.results]
        except Exception:
            all_properties = []  # Will fetch with default properties

        # Paginate
        records: list[dict] = []
        after: str | None = None
        page_size = min(100, max_records)

        try:
            api = self.client.crm.objects.basic_api
            while len(records) < max_records:
                kwargs: dict = {"object_type": object_name, "limit": page_size}
                if all_properties:
                    kwargs["properties"] = all_properties
                if after:
                    kwargs["after"] = after

                page = api.get_page(**kwargs)

                for item in page.results:
                    records.append(item.properties if hasattr(item, "properties") else {})

                if page.paging and page.paging.next and page.paging.next.after:
                    after = page.paging.next.after
                else:
                    break

        except Exception as exc:
            err_msg = str(exc)
            if "429" in err_msg or "RATE_LIMIT" in err_msg.upper():
                raise CRMRateLimitError("Limite de requests de HubSpot excedido. Espera unos minutos.") from exc
            if "401" in err_msg:
                raise CRMAuthError("Token de HubSpot expirado. Reconecta.") from exc
            raise CRMConnectionError(f"Error al consultar '{object_name}': {exc}") from exc

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        return df.head(max_records)
