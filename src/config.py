from typing import Annotated, ClassVar, List

from pydantic import BeforeValidator
from pydantic_settings import (
    BaseSettings,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)
from typing_extensions import Any


def convert_str2int(v: Any):
    if isinstance(v, str):
        v = int(v)
    return v


def convert_str2float(v: Any):
    if isinstance(v, str):
        v = float(v)
    return v


FloatMapStr = Annotated[float, BeforeValidator(convert_str2float)]
IntMapStr = Annotated[int, BeforeValidator(convert_str2int)]


class AppSettings(BaseSettings):
    path_to_modelfile: str
    feature_names: List[str]

    model_config: ClassVar[SettingsConfigDict] = SettingsConfigDict(
        json_file=(
            "config.json",
            "config_debug.json",
        ),
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            JsonConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


app_config = AppSettings()  # pyright: ignore[reportCallIssue]
