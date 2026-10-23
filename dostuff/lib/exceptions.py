class ConfirmationRequired(Exception):
    """Legacy approval signal, kept only for backwards compatibility.

    Tools no longer raise this: approval is decided by the permission layer in
    `dostuff.agent.permissions` before a tool is dispatched.
    """

    def __init__(self, message: str, resume_args: dict):
        super().__init__(message)
        self.message = message
        self.resume_args = resume_args


class PathOutsideProject(ValueError):
    """Raised when a tool path resolves outside the project root."""


class PathAccessDenied(PermissionError):
    """Raised when a tool path is on the always-denied list (secrets, harness data)."""
