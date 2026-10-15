"""`uv run python -m app.worker`: worker como proceso aparte (ver `app.worker.worker._main`)."""

import asyncio
import contextlib

from app.worker.worker import _main

with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(_main())
