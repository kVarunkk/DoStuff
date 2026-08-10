# DoStuff

## AI Agent Harness

<img width="3199" height="2009" alt="image" src="https://github.com/user-attachments/assets/8057e0d8-6b43-49f4-8022-6fa752f6131b" />


- reAct tool calling
- opentelemetry tracing
- short term memory
- long term memory
  - Procedural
  - Semantic
  - Episodic
- context management
- skill support
- mcp servers support (stdio and http (Dynamic Client Registration support only))
- self-learning loop

## Setup

- run `git clone https://github.com/kVarunkk/DoStuff.git`
- create a `.env` in the root and refer `.env.example` for the variables
- create and activate a `venv`
- run `pip install -r requirements.txt`
- create a `DOSTUFF.md` for giving your agent a personality. Take inspiration from `DOSTUFF.sample.md`
- run `python app.py`

## Tools

- `write_file`
- `read_file`
- `list_files`
- `delete_file`
- `run_code`

## Guidelines

- Main agent code is present in `agent/run_agent.py`
- To create a new tool:
  - add a new file in the `tools` directory with name matching that of the tool func
  - update `tools/definitions.py`
- New files will be created in the `agent_workspace` directory
- Add skills in the `skill` directory. Sample skills present. Refer [this](https://agentskills.io/home) for more info.

### Short Term Memory

- Uses SQLite for Storage, In-Memory Storage also available.

### Long Term Memory

- Procedural: Saved as skills. Learning loop creates/updates skills at the end of the session.
- Semantic: Saved in Vector DB. User scoped. (To test, use the same user id)
- Episodic: Saved in Vector DB. User and Session scoped. (To test, use the same user id and session id)

### MCP Servers

- Create and add mcp servers in `mcp_config.json`. Sample remote and local servers are present in `mcp_config_sample.json`.

### Self-learning Loop

- At the end of each session on exit, a seperate agent loop decides if there is something in the conversation history worth learning from. If present, the agent creates a new skill for that and updates if the skill is already present using the `skill-creator` skill present in the `/skills` directory.

### Observability

- Use this command to spin up a Jaeger container:
  ``docker run -d --name jaeger `
-e SPAN_STORAGE_TYPE=badger `
-e BADGER_EPHEMERAL=false `
-e BADGER_DIRECTORY_VALUE=/badger/data `
-e BADGER_DIRECTORY_KEY=/badger/key `
-v ${PWD}/jaeger_data:/badger `
-p 16686:16686 `
-p 4317:4317 `
-p 4318:4318 `
jaegertracing/all-in-one:latest``
- Open `http://localhost:16686` on your browser to view live traces.
