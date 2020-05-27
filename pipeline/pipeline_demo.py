import time
import multiprocessing


def process(pipe_in, pipe_out, lock, namespace):
    while not namespace.quit_event.is_set or any(pipe.poll() for pipe in namespace.pipes_out):
        with lock:
            v = pipe_in.recv()
        print(multiprocessing.current_process(), v)
        if pipe_out is not None:
            pipe_out.send(v)
        time.sleep(1)


if __name__ == "__main__":
    p1_out, p1_in = multiprocessing.Pipe(duplex=False)
    p2_out, p2_in = multiprocessing.Pipe(duplex=False)
    p3_out, p3_in = multiprocessing.Pipe(duplex=False)

    p1_lock = multiprocessing.Lock()
    p2_lock = multiprocessing.Lock()
    p3_lock = multiprocessing.Lock()

    manager = multiprocessing.Manager()
    namespace = manager.Namespace()
    namespace.pipes_in = [p1_in, p2_in, p3_in]
    namespace.pipes_out = [p1_out, p2_out, p3_out]
    namespace.quit_event = manager.Event()

    pipeline = [
        multiprocessing.Process(target=process, args=(p1_out, p2_in, p1_lock, namespace)),
        multiprocessing.Process(target=process, args=(p2_out, p3_in, p2_lock, namespace)),
        multiprocessing.Process(target=process, args=(p3_out, None, p3_lock, namespace)),
        multiprocessing.Process(target=process, args=(p3_out, None, p3_lock, namespace)),
    ]

    for pipe in pipeline:
        pipe.start()

    values = "Hallo Welt!"
    for v in values:
        # v = p1_in.recv()
        print(multiprocessing.current_process(), v)
        p1_in.send(v)
        time.sleep(0.5)
    namespace.quit_event.set()

    for pipe in pipeline:
        pipe.join()
