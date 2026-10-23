import asyncio
import json
from dostuff.lib.memory.chroma import get_chroma_client
from typing import List, Dict, Any


class MCPToolRegistryStore:
    def __init__(self, persist_path: str = ".dostuff/data/chroma"):
        self._client = get_chroma_client(persist_path)
        self._collection = self._client.get_or_create_collection(
            name="mcp_tools_index",
            metadata={"hnsw:space": "cosine"}
        )

    async def register_tools(self, server_name: str, tools: List[Any]) -> None:
        """Replaces this server's indexed tools with `tools` (drops tools it no longer lists)."""
        def _sync_replace():
            self._collection.delete(where={"server_name": server_name})
            if not tools:
                return
            ids = []
            documents = []
            metadatas = []
            for tool in tools:
                ids.append(f"{server_name}:{tool.name}")
                # Document string optimized for semantic vector lookup
                documents.append(
                    f"Tool Name: {tool.name}\nDescription: {tool.description}\nInputs: {json.dumps(tool.input_schema)} \n"
                    f"Outputs: {json.dumps(tool.output_schema) if tool.output_schema else 'None'}"
                )
                metadatas.append({
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "description": tool.description or "",
                    # Store schema as JSON string for easy reconstruction
                    "input_schema_json": json.dumps(tool.input_schema),
                    "output_schema_json": json.dumps(tool.output_schema) if tool.output_schema else None
                })
            # ids are unique per server; a tool listed twice keeps its last definition
            unique = {i: (d, m) for i, d, m in zip(ids, documents, metadatas)}
            self._collection.upsert(
                ids=list(unique),
                documents=[d for d, _ in unique.values()],
                metadatas=[m for _, m in unique.values()],
            )

        await asyncio.to_thread(_sync_replace)

    async def search_tools(self, user_intent: str, top_k: int = 5, server_names: list[str] | None = None) -> List[Dict[str, Any]]:
        """Finds the most relevant tools for a given intent, only among `server_names` when given."""
        if server_names is not None and not server_names:
            return []

        def _sync_query():
            n = min(top_k, self._collection.count())
            if n <= 0:
                return {}
            kwargs: dict[str, Any] = {"query_texts": [user_intent], "n_results": n}
            if server_names is not None:
                kwargs["where"] = {"server_name": {"$in": list(server_names)}}
            return self._collection.query(**kwargs)

        results = await asyncio.to_thread(_sync_query)

        raw_metas = results.get("metadatas") or [[]]
        metadatas = raw_metas[0] if raw_metas and raw_metas[0] is not None else []

        matched_tools = []
        for meta in metadatas:
            input_schema_json = meta.get("input_schema_json")
            output_schema_json = meta.get("output_schema_json")

            input_schema = json.loads(input_schema_json) if isinstance(input_schema_json, str) else input_schema_json
            output_schema = json.loads(output_schema_json) if isinstance(output_schema_json, str) else output_schema_json

            matched_tools.append({
                "name": meta["tool_name"],
                "server_name": meta["server_name"],
                "description": meta["description"],
                "input_schema": input_schema,
                "output_schema": output_schema
            })

        return matched_tools
