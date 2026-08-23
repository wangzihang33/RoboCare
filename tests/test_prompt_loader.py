from unittest import TestCase

from utils.prompt_loader import load_system_prompts


class RoutePromptTests(TestCase):
    def test_route_prompt_keeps_only_allowed_tool_descriptions(self):
        prompt = load_system_prompts(("rag_summarize",))

        self.assertIn("rag_summarize", prompt)
        self.assertNotIn("get_weather", prompt)
        self.assertNotIn("fetch_external_data", prompt)
        self.assertNotIn("web_search", prompt)
        self.assertIn("### 核心准则", prompt)
        self.assertIn("### 输出规则", prompt)

    def test_route_without_tools_removes_tool_section(self):
        prompt = load_system_prompts(())

        self.assertNotIn("### 可使用工具及能力边界", prompt)
        self.assertNotIn("rag_summarize", prompt)
        self.assertIn("### 输出规则", prompt)

    def test_report_route_keeps_report_constraint_and_tools(self):
        prompt = load_system_prompts(
            ("fetch_external_data", "fill_context_for_report", "rag_summarize")
        )

        self.assertIn("报告生成强约束", prompt)
        self.assertIn("fetch_external_data", prompt)
        self.assertIn("fill_context_for_report", prompt)
        self.assertIn("rag_summarize", prompt)
        self.assertNotIn("web_search", prompt)
