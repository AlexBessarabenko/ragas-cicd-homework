"""
Совместимость ragas 0.2.x с langchain-community >= 0.4.

Ragas 0.2.x импортирует ChatVertexAI/VertexAI из старых путей
langchain_community.chat_models.vertexai и langchain_community.llms.vertexai,
которые были удалены в новых версиях langchain-community.

Этот модуль создаёт минимальные заглушки для этих модулей до первого
импорта ragas, чтобы избежать ModuleNotFoundError.
"""
import sys
import types


def _install_stubs() -> None:
    if "langchain_community.chat_models.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[misc]
        sys.modules["langchain_community.chat_models.vertexai"] = mod

    if "langchain_community.llms.vertexai" not in sys.modules:
        mod = types.ModuleType("langchain_community.llms.vertexai")
        mod.VertexAI = type("VertexAI", (), {})  # type: ignore[misc]
        sys.modules["langchain_community.llms.vertexai"] = mod


_install_stubs()
