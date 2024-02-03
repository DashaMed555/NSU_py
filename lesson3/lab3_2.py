class Queue(list):
    def __init__(self, queue=None):
        if queue is None or not isinstance(queue, list):
            queue = []
        super().__init__(queue)


class Stack(list):
    def __init__(self, stack=None):
        if stack is None or not isinstance(stack, list):
            stack = []
        super().__init__(stack)


class QueueContainer:
    def __init__(self, queue=None):
        if isinstance(queue, Queue):
            self._queue = queue
        else:
            self._queue = []

    def push(self, el):
        self._queue.append(el)

    def pop(self):
        if len(self._queue):
            return self._queue.pop(0)
        else:
            print("The queue is empty.")
            return None

    def get_queue(self):
        return Queue(self._queue)


def test():
    q = Queue([1, 2, 3])
    qc = QueueContainer(q)
    print(qc.get_queue())
    qc.push("New")
    qu = qc.get_queue()
    print(f"qu = {qu}")
    qu.append("Smth secret...")
    print(f"qu = {qu}")
    print(f"But queue = {qc.get_queue()}")
    print(f"Pop: {qc.pop()}")
    print(qc.get_queue())
    print(f"Pop: {qc.pop()}")
    print(qc.get_queue())
    print(f"Pop: {qc.pop()}")
    print(qc.get_queue())
    print(f"Pop: {qc.pop()}")
    print(qc.get_queue())
    print(f"Pop: {qc.pop()}")
    print(qc.get_queue())


test()
