from dataclasses import dataclass
from typing import Optional


@dataclass
class GlobalConfig:
    openai_api_base: Optional[str] = ""
    openai_api_key: Optional[str] = ""

    auth0_client_id: Optional[str] = ""
    auth0_domain: Optional[str] = ""

    myscale_user: Optional[str] = ""
    myscale_password: Optional[str] = ""
    myscale_host: Optional[str] = ""
    myscale_port: Optional[int] = 443

    query_model: Optional[str] = ""
    chat_model: Optional[str] = ""

    untrusted_api: Optional[str] = ""
    myscale_enable_https: Optional[bool] = True
