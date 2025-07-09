import time
from functools import wraps
import numpy as np

def latency_benchmark(warmup=5, repeat=50):
    def decorator(func):
        @wraps(func)
        def wrapper():
            infer_once = func()

            print("Warm Up")
            for _ in range(warmup):
                infer_once()

            print("Benchmark")
            start = time.time()
            for _ in range(repeat):
                infer_once()
            end = time.time()

            latency = (end - start) / repeat * 1000
            throughput = repeat / (end - start)

            print(f"Benchmark Results: ")
            print(f"Latency: {latency:.2f} ms")
            print(f"Throughput: {throughput:.2f} FPS")
        return wrapper
    return decorator

def throughput_benchmark(warmup=10, repeat=100):
    def decorator(func):
        @wraps(func)
        def wrapper():
            infer_requests, input_data, input_name = func()
            num_requests = len(infer_requests)

            def do_infer():
                for i in range(num_requests):
                    infer_requests[i].start_async({input_name: input_data[i]})
                for req in infer_requests:
                    req.wait()

            print("Warm Up")
            for _ in range(warmup):
                do_infer()

            print("Benchmark")
            latencies = []
            for _ in range(repeat):
                start = time.time()
                do_infer()
                end = time.time()
                latencies.append(end - start)

            latency = np.mean(latencies) * 1000
            throughput = num_requests / np.mean(latencies)

            print(f"Benchmark Results: ")
            print(f"Latency: {latency:.2f} ms")
            print(f"Throughput: {throughput:.2f} FPS")
        return wrapper
    return decorator

