"""The optional extras (`pip install "dostuff[...]"`): which are installed, and what each one adds.

`dostuff doctor`, the end of `dostuff setup` and /tools use this so a plain `pip install dostuff` user learns
what they are missing without reading the README. `memory` is not here: doctor already reports it in detail.
"""
import importlib.util
from dataclasses import dataclass


@dataclass(frozen=True)
class Extra:
    name: str
    modules: tuple[str, ...]  # importable names; the extra counts as installed when all are present
    adds: str  # what the user gets, for a line like "search: not installed (<adds>)"
    without: str  # what happens without it


EXTRAS: tuple[Extra, ...] = (
    Extra("search", ("ddgs",), "steadier web_search through several engines",
          "web_search reads DuckDuckGo's HTML page, which rate-limits"),
    Extra("docs", ("pypdf", "docx", "openpyxl", "pptx"), "Word, Excel, PowerPoint and PDF files as text",
          "@path and read_file can't read .docx/.xlsx/.pptx, or PDFs on models that can't read them"),
    Extra("images", ("PIL",), "large images downscaled to fit the model's limit",
          "an image over the model's limit is refused instead of resized"),
)


def install_hint(*names: str) -> str:
    return f'pip install "dostuff[{",".join(names)}]"'


def extra_installed(extra: Extra) -> bool:
    try:
        return all(importlib.util.find_spec(module) is not None for module in extra.modules)
    except (ImportError, ValueError):
        return False


def missing_extras() -> list[Extra]:
    return [extra for extra in EXTRAS if not extra_installed(extra)]
