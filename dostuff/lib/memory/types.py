from typing import Literal

EpisodeType = Literal["DECISION", "PREFERENCE", "CORRECTION", "EVENT", "LEARNING"]

MemorySource = Literal["user_message", "assistant"]
