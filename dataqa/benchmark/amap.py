import asyncio


async def amap(async_function, iterable, limit=None):
    """
    Asynchronously maps an async function over an iterable, with optional concurrency limit.

    Args:
        async_function: The async function to apply.
        iterable: The iterable of arguments.
        limit: The maximum number of concurrent tasks (optional).

    Returns:
        A list of results from the async function calls, in the original order.
    """

    semaphore = asyncio.Semaphore(limit) if limit else None
    tasks = []
    results = [None] * len(iterable)  # Pre-allocate results to maintain order

    async def _worker(index, arg):
        async with (
            semaphore if semaphore else async_noop()
        ):  # Acquire semaphore if limit set
            results[index] = await async_function(
                arg
            )  # Store result at the correct index

    for index, arg in enumerate(iterable):
        tasks.append(_worker(index, arg))  # Create worker tasks

    await asyncio.gather(*tasks)  # Run all tasks concurrently
    return results


class async_noop:
    """Dummy context manager for when no limit is set"""

    async def __aenter__(self):
        return

    async def __aexit__(self, *args):
        return
