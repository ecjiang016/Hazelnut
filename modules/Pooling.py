class GlobalPooling:
    def __init__(self):
        self.np = None

    def Forward(self, inp):
        return inp.sum(axis=(2, 3)) / self.HW

    def Forward_training(self, inp):
        return inp.sum(axis=(2, 3)) / self.HW

    def Backward(self, inp):
        return inp[:, :, None, None] * self.ones[None, :, :, :] / self.HW

    def Build(self, shape):
        _, C, H, W = shape
        self.HW = H * W
        self.ones = self.np.ones((C, H, W))

    def Save(self):
        return {'args':(), 'var':(self.HW, self.ones)}

    def Load(self):
        self.HW, self.ones == 'var'