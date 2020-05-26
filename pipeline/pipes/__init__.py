import abc


class Target(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def __call__(self, document):
        pass


def worker(target: Target, pipe_in, pipe_out, lock):
    while True:
        with lock:
            try:
                if hasattr(pipe_in, "recv"):
                    value = pipe_in.recv()
                else:
                    value = pipe_in.get()
            except EOFError:
                break
        result = target(value)
        if pipe_out is not None:
            pipe_out.send(result)

