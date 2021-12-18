import numpy as np

# Special structure for storing and displaying experimental results
class Recorder:
    def __init__(self, outputdir):
        self.meter = {}
        self.dir = outputdir
    def __getitem__(self, key):
        return self.meter[key]
    def add_key(self, key):
        if key not in self.meter.keys():
            self.meter[key] = []
    def summarize(self):
        f = open(self.dir, 'w')
        for key in self.meter.keys():
            l = self.meter[key]
            self.meter[key] = np.mean(np.array(l))
            f.write(key + ', ' + str(self.meter[key]) + '\n')
        f.close()
