from gunpowder.producer_pool import ProducerPool
import unittest
import multiprocessing

class PredictWorkers:

    def __init__(self, n):
        '''Called by main process. Spawns n workers.'''

        self.workers = ProducerPool([lambda gpu=gpu: self.__predict_with_gpu(gpu) for gpu in range(n)])
        self.in_queue = multiprocessing.Queue(maxsize=n)
        self.manager = multiprocessing.Manager()

        # process local
        self.out_queue = None

    def setup(self):
        '''Called by main process.'''

        self.workers.start()

    def teardown(self):
        '''Called by main process.'''

        self.workers.stop()

    def predict(self, batch):
        '''Called by subprocesses. Send batch to workers, wait for result on 'out_queue'.'''

        # create a process local output queue
        if self.out_queue is None:
            self.out_queue = self.manager.Queue(maxsize=1)

        self.in_queue.put((batch, self.out_queue))
        result = self.out_queue.get()

        # workers always produce something, even though we redirected their 
        # result to out_queue, pop it here
        # assert self.workers.get() == 0

        return result

    def __predict_with_gpu(self, gpu):

        while True:

            print("PredictWorker " + str(gpu) + " waiting for batch...")
            (batch, out_queue) = self.in_queue.get()
            print("PredictWorker " + str(gpu) + " got batch " + str(batch))

            prediction = "prediction of " + str(batch)
            out_queue.put(prediction)

class SharedQueueTest(unittest.TestCase):

    def test_output(self):

        print("Creating predict workers...")
        self.predict_workers = PredictWorkers(3)

        print("Setting up predict workers...")
        self.predict_workers.setup()

        # create k batch request workers
        request_workers = ProducerPool([lambda i=i: self.request_batches(i) for i in range(5)], queue_size=10)
        request_workers.start()
        for i in range(25):
            batch = request_workers.get()
            print("Main: Got result " + str(batch))
        request_workers.stop()

        self.predict_workers.teardown()

    def request_batches(self, i):

        for j in range(5):
            print("RequestWorker " + str(i) + ": Sending work to predict workers...")
            r = self.predict_workers.predict(str((i,j)))
            print("RequestWorker " + str(i) + ": Got result " + str(r))
