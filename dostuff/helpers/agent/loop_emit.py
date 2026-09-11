async def loop_emit(et, data, adapter=None):
        if adapter and hasattr(adapter, "emit"):
            try:
                adapter.emit(et, data)
            except Exception:
                pass