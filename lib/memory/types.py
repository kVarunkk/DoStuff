from pydantic import BaseModel, Field
from typing import List, Literal

class TurnContext(BaseModel):
    role: str = Field(description="Role of the turn, e.g., 'user' or 'model'")
    text: str = Field(description="Text content of the conversational step/turn")

class AtomicEpisode(BaseModel):
    event_type: Literal["DECISION", "PREFERENCE", "CORRECTION", "EVENT", "LEARNING"] = Field(
        description="Category of the key moment."
    )
    anchor_event: str = Field(
        description="A concise 1-2 sentence description of the core event/action to be embedded."
    )
    summary: str = Field(
        description="Detailed reflection on what happened, why, and the outcome."
    )
    tags: List[str] = Field(
        default_factory=list, 
        description="Key topics or technologies involved (e.g. ['sqlite', 'wal_mode'])"
    )
    context_window: List[TurnContext] = Field(
        description="The 2-3 surrounding conversational turns framing this moment."
    )

class SessionEpisodicExtraction(BaseModel):
    episodes: List[AtomicEpisode] = Field(
        description="List of atomic episodic memories extracted from the conversation."
    )

class SemanticMemory(BaseModel):
    key: str = Field(description="Name of memory")
    value: str = Field(description="Value of memory")

class SessionSemanticMemoriesExtraction(BaseModel):
    facts: List[SemanticMemory] = Field(description="List of semantic memories extracted from conversation.")    