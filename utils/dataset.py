import numpy as np

class dataloader:
    """
    An iterator that returns `input_batch`, `label_batch`, and `current_epoch`
    """
    def __init__(self, batch_size:int, epochs:int, inp, labels):
        """
        The first dimension shape of input (inp) must match the first dimension shape of labels
        """
        self.batch_size = batch_size
        self.epochs = epochs
        self.inp = inp
        self.labels = labels

        assert self.inp.shape[0] == self.labels.shape[0], f"Shape mismatch. inp shape: {self.inp.shape}, labels shape: {self.labels.shape}. array.shape[0] for inp and label must match."

        self.dataset_size = self.labels.shape[0]
        self.batches_in_epoch = self.dataset_size // batch_size

        self.current_batch_num = 0 #Goes from 0 to batches_in_epoch
        self.current_epoch = 0

        self.random_array = np.arange(self.batches_in_epoch * self.batch_size)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self.current_batch_num % self.batches_in_epoch == 0:
            np.random.shuffle(self.random_array)

            self.current_batch_num = 0
            self.current_epoch += 1

        slice_indices = self.random_array[self.current_batch_num * self.batch_size : (self.current_batch_num + 1) * self.batch_size]
        input_batch = self.inp[slice_indices]
        label_batch = self.labels[slice_indices]

        self.current_batch_num += 1

        if self.current_epoch == self.epochs:
            raise StopIteration

        return input_batch, label_batch, self.current_epoch