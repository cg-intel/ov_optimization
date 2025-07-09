import time
from functools import wraps

def benchmark(warmup=5, repeat=50):
    def decorator(func):
        @wraps(func)
        def wrapper():
            infer_once = func()

            for _ in range(warmup):
                infer_once()

            start = time.time()
            for _ in range(repeat):
                infer_once()
            end = time.time()

            latency = (end - start) / repeat * 1000
            throughput = repeat / (end - start)

            print(f"Latency: {latency:.2f} ms")
            print(f"Throughput: {throughput:.2f} FPS")
        return wrapper
    return decorator
