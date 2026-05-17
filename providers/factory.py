from providers.base_provider import BaseProvider
from config import settings


def get_provider() -> BaseProvider:
    """
    Returns the correct provider based on settings.PROVIDER.
    """
    provider_name = settings.PROVIDER.lower().strip()

    if provider_name == "anthropic":
        from providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider()

    elif provider_name == "openai":
        from providers.openai_provider import OpenAIProvider

        return OpenAIProvider()

    else:
        raise ValueError(
            f"Unknown provider: '{provider_name}'. "
            f"Valid options are: 'anthropic', 'openai'"
        )
