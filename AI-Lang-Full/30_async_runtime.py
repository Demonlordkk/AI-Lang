"""Phase 30: small structured async runtime boundary."""
import asyncio
class AsyncRuntime:
    def run(self, awaitable):
        return asyncio.run(awaitable)
    async def gather(self, *awaitables):
        return await asyncio.gather(*awaitables)
