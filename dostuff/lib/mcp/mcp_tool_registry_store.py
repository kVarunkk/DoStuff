import asyncio
import json
import chromadb
from typing import List, Dict, Any

class MCPToolRegistryStore:
    def __init__(self, persist_path: str = ".dostuff/data/chroma"):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            name="mcp_tools_index", 
            metadata={"hnsw:space": "cosine"}
        )

    async def register_tools(self, server_name: str, tools: List[Any]) -> None:
        """Indexes or updates tool schemas from an MCP server."""
        if not tools:
            return

        def _sync_upsert():
            ids = []
            documents = []
            metadatas = []

            for tool in tools:
                tool_id = f"{server_name}:{tool.name}"
                
                # Document string optimized for semantic vector lookup
                doc_text = f"Tool Name: {tool.name}\nDescription: {tool.description}\nInputs: {json.dumps(tool.input_schema)} \nOutputs: {json.dumps(tool.output_schema) if tool.output_schema else 'None'}"
                
                ids.append(tool_id)
                documents.append(doc_text)
                metadatas.append({
                    "server_name": server_name,
                    "tool_name": tool.name,
                    "description": tool.description or "",
                    # Store schema as JSON string for easy reconstruction
                    "input_schema_json": json.dumps(tool.input_schema),
                    "output_schema_json": json.dumps(tool.output_schema) if tool.output_schema else None
                })

            self._collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

        await asyncio.to_thread(_sync_upsert)

    async def search_tools(self, user_intent: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Finds the most relevant tools for a given user prompt or step intent."""
        def _sync_query():
            return self._collection.query(
                query_texts=[user_intent],
                n_results=top_k
            )

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