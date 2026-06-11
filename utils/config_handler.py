from utils.path_tool import get_abs_path
import os
import yaml


def load_env_file(env_path: str = get_abs_path(".env"), encoding: str = "utf-8"):
    if not os.path.exists(env_path):
        return

    with open(env_path, "r", encoding=encoding) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and not os.environ.get(key):
                os.environ[key] = value


def apply_env_overrides(config: dict, mapping: dict[str, str]):
    for config_key, env_key in mapping.items():
        env_value = os.getenv(env_key)
        if env_value is not None:
            config[config_key] = env_value
    return config


load_env_file()

def load_rag_config(config_path: str=get_abs_path("config/rag.yml"), encoding: str="utf-8"):
    with open(config_path, 'r', encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_chroma_config(config_path: str=get_abs_path("config/chroma.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_prompts_config(config_path: str=get_abs_path("config/prompts.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_agent_config(config_path: str=get_abs_path("config/agent.yml"), encoding: str="utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return apply_env_overrides(config, {
        "amap_key": "AMAP_KEY",
        "amap_weather_api_url": "AMAP_WEATHER_API_URL",
        "external_data_db_path": "USER_USAGE_DB_PATH",
        "external_data_path": "USER_USAGE_SEED_CSV_PATH",
    })
    
def load_websearch_config(config_path: str = get_abs_path("config/websearch.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return apply_env_overrides(config, {
        "serper_api_url": "SERPER_API_URL",
        "serper_api_key": "SERPER_API_KEY",
        "openai_api_key": "OPENAI_API_KEY",
    })


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()
websearch_conf = load_websearch_config()

if __name__ == "__main__":
    print(rag_conf["chat_model_name"])
