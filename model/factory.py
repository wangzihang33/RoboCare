from abc import ABC, abstractmethod
import os
from typing import Optional
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_conf


DEFAULT_DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def get_dashscope_api_key() -> str:
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise ValueError("缺少 DASHSCOPE_API_KEY，请在项目 .env 中配置后重新运行")
    return api_key

class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass
    

class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return build_chat_model(
            model_name=rag_conf.get("chat_model_name"),
            provider=str(rag_conf.get("chat_model_provider", "dashscope")),
            api_key_env=rag_conf.get("chat_model_api_key_env") or None,
            base_url=rag_conf.get("chat_model_base_url") or None,
        )


def build_chat_model(
    model_name: str | None = None,
    *,
    provider: str = "dashscope",
    api_key_env: str | None = None,
    base_url: str | None = None,
) -> ChatOpenAI:
    normalized_provider = provider.strip().lower()
    if normalized_provider == "deepseek":
        key_env = api_key_env or "DEEPSEEK_API_KEY"
        api_key = os.getenv(key_env)
        if not api_key:
            raise ValueError(f"缺少 {key_env}，无法初始化 DeepSeek 模型")
        resolved_base_url = base_url or os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        )
        resolved_model_name = model_name or "deepseek-chat"
    else:
        key_env = api_key_env or "DASHSCOPE_API_KEY"
        api_key = os.getenv(key_env) or get_dashscope_api_key()
        resolved_base_url = base_url or os.getenv(
            "DASHSCOPE_BASE_URL", DEFAULT_DASHSCOPE_BASE_URL
        )
        resolved_model_name = model_name or rag_conf["chat_model_name"]

    return ChatOpenAI(
        model=resolved_model_name,
        api_key=api_key,
        base_url=resolved_base_url,
        temperature=0,
    )
    

class EmbeddingModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            dashscope_api_key=get_dashscope_api_key(),
        )
    

chat_model = ChatModelFactory().generator()
embed_model = EmbeddingModelFactory().generator()
