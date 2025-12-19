from langchain.tools import tool

from reviewbot.infra.embeddings.store_manager import CodebaseStoreManager
from reviewbot.context import store_manager_ctx


@tool
def compile_codebase() -> str:
    """Compile the codebase to see if it compiles successfully.

    Returns:
        string with the results of the compilation
    """
    store = store_manager_ctx.get().get_store()

    results = store.compile("go build .")
    if not results:
        return "Compilation failed."

    return results
