from collections import defaultdict
import asyncio, json, time

class ProgressBroker:
    def __init__(self):
        self._subs = defaultdict(set)  # key(meeting_dir) -> set[asyncio.Queue]

    async def subscribe(self, key: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self._subs[key].add(q)
        return q

    async def unsubscribe(self, key: str, q: asyncio.Queue):
        self._subs[key].discard(q)

    async def publish(self, key: str, event: dict):
        # 자동 타임스탬프
        event.setdefault("ts", time.time())
        for q in list(self._subs[key]):
            await q.put(event)