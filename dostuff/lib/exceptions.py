class ConfirmationRequired(Exception):
    """Raised by a tool to signal it needs explicit user approval before proceeding.

    The harness catches this, prompts the user, and — if approved — re-invokes the
    tool with resume_args merged into the original arguments.
    """

    def __init__(self, message: str, resume_args: dict):
        super().__init__(message)
        self.message = message
        self.resume_args = resume_args