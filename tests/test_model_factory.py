from unittest import TestCase
from unittest.mock import patch

import model.factory as factory


class ChatModelFactoryTests(TestCase):
    def test_main_chat_model_uses_configured_provider(self):
        configured_values = {
            "chat_model_provider": "deepseek",
            "chat_model_name": "deepseek-v4-pro",
            "chat_model_api_key_env": "MAIN_DEEPSEEK_API_KEY",
            "chat_model_base_url": "https://api.deepseek.com",
        }
        captured = {}

        def fake_build_chat_model(
            model_name=None,
            *,
            provider="dashscope",
            api_key_env=None,
            base_url=None,
        ):
            captured.update(
                model_name=model_name,
                provider=provider,
                api_key_env=api_key_env,
                base_url=base_url,
            )
            return "configured-model"

        with patch.dict(factory.rag_conf, configured_values):
            with patch.object(factory, "build_chat_model", fake_build_chat_model):
                model = factory.ChatModelFactory().generator()

        self.assertEqual(model, "configured-model")
        self.assertEqual(
            captured,
            {
                "model_name": "deepseek-v4-pro",
                "provider": "deepseek",
                "api_key_env": "MAIN_DEEPSEEK_API_KEY",
                "base_url": "https://api.deepseek.com",
            },
        )
