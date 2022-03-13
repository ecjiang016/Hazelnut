class SkipConnClass:
    """
    Skip connection
    """
    def __init__(self):
        self.start = True
    
    def Forward(self, inp):
        if self.start:
            self.start = not self.start
            self.cache = inp.copy()
            return inp
        
        else:
            self.start = not self.start
            return inp + self.cache

    def Forward_training(self, inp):
        if self.start:
            self.start = not self.start
            self.cache = inp.copy()
            return inp
        
        else:
            self.start = not self.start
            return inp + self.cache

    def Backward(self, inp):
        if self.start:
            self.start = not self.start
            self.cache = inp.copy()
            return inp
        
        else:
            self.start = not self.start
            return inp + self.cache

    def Build(self, _):
        self.start = True

    def Save(self):
        return {'args':(), 'var':()}

    def Load(self):
        self.start = True

skip = SkipConnClass()

def SkipConn(_) -> SkipConnClass:
    """
    Skip connection

    Replacement for the skip connection class. Use this one instead of that one when making a neural net.
    """
    return skip