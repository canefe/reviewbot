from langchain.tools import tool

from reviewbot.infra.embeddings.in_memory_store import get_store


@tool
def compile_codebase() -> str:
    """Compile the codebase to see if it compiles successfully.

    Returns:
        string with the results of the compilation
    """
    store = get_store()

    results = store.compile("go build .")
    if not results:
        return "Compilation failed."

    return results
