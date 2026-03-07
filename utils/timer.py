from time import time


class Timer:
    def __init__(self):
        self.start_time = time()
        self.l = []
        
    def start(self):
        self.start_time = time()
    
    def stop(self):
        self.l.append(time() - self.start_time)
        return self.l[-1]
    
    def avg(self):
        if len(self.l) == 0:
            return 0
        return sum(self.l) / len(self.l)
    
    def clear(self):
        self.l = []