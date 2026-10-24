"""Import RAGAS 0.4 alongside LangChain 1.x.

ragas 0.4.3 imports `langchain_community.chat_models.vertexai.ChatVertexAI`, which langchain-community
0.4 removed. RAGAS only uses it in an isinstance() check for Vertex models (which we don't use), so a
placeholder class is registered before RAGAS is imported. Remove this once RAGAS drops the import.
"""

import sys
import types

_MODULE = "langchain_community.chat_models.vertexai"


def install() -> None:
    if _MODULE in sys.modules:
        return
    try:
        __import__(_MODULE)
    except ImportError:
        shim = types.ModuleType(_MODULE)
        shim.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[_MODULE] = shim
