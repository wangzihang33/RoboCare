import re

from utils.config_handler import prompts_conf
from utils.path_tool import get_abs_path
from utils.logger_handler import logger


_TOOL_SECTION_HEADING = "### 可使用工具及能力边界"
_OUTPUT_SECTION_HEADING = "### 输出规则"
_TOOL_HEADING_PATTERN = re.compile(r"(?m)^\d+\.\s+([A-Za-z_]\w*)：")
_REPORT_RULE_PATTERN = re.compile(r"(?m)^5\.\s*报告生成强约束：.*\n")


def load_system_prompts(tool_names=None):
    try:
        system_prompt_path = get_abs_path(prompts_conf["main_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_system_prompts]在yaml配置项中没有main_prompt_path配置项")
        raise e
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as prompt_file:
            prompt = prompt_file.read()
        if tool_names is None:
            return prompt
        return _filter_system_tool_sections(prompt, tool_names)
    except Exception as e:
        logger.error(f"[load_system_prompts]在加载系统提示词时发生错误: {str(e)}")
        raise e


def _filter_system_tool_sections(prompt: str, tool_names) -> str:
    """Keep the shared prompt and only the tools exposed to this route."""
    section_start = prompt.find(_TOOL_SECTION_HEADING)
    output_start = prompt.find(_OUTPUT_SECTION_HEADING, section_start + 1)
    if section_start < 0 or output_start < 0:
        return prompt

    allowed = {str(name) for name in tool_names}
    shared_prompt = prompt[:section_start]
    if not {"fetch_external_data", "fill_context_for_report"}.issubset(allowed):
        shared_prompt = _REPORT_RULE_PATTERN.sub("", shared_prompt)
    tool_section = prompt[section_start:output_start].strip()
    matches = list(_TOOL_HEADING_PATTERN.finditer(tool_section))
    selected_sections: list[str] = []
    for index, match in enumerate(matches):
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(tool_section)
        if match.group(1) in allowed:
            selected_sections.append(tool_section[match.start():section_end].strip())

    if selected_sections:
        filtered_tools = f"{_TOOL_SECTION_HEADING}\n\n" + "\n\n".join(selected_sections)
        return (
            shared_prompt.rstrip()
            + "\n\n"
            + filtered_tools
            + "\n\n"
            + prompt[output_start:].lstrip()
        )

    return shared_prompt.rstrip() + "\n\n" + prompt[output_start:].lstrip()
    
def load_rag_prompts():
    try:
        rag_prompt_path = get_abs_path(prompts_conf["rag_summarize_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_rag_prompts]在yaml配置项中没有mrag_summarize_prompt_path配置项")
        raise e
    try:
        return open(rag_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_rag_prompts]在加载rag提示词时发生错误: {str(e)}")
        raise e
    

def load_report_prompts():
    try:
        report_prompt_path = get_abs_path(prompts_conf["report_prompt_path"])
    except KeyError as e:
        logger.error(f"[load_report_prompts]在yaml配置项中没有report_prompt_path配置项")
        raise e
    try:
        return open(report_prompt_path, "r", encoding="utf-8").read()
    except Exception as e:
        logger.error(f"[load_report_prompts]在加载报告生成提示词时发生错误: {str(e)}")
        raise e
    

if __name__ == "__main__":
    # print(load_system_prompts())
    # print(load_rag_prompts())
    print(load_report_prompts())
