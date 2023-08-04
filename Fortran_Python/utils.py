class prep():
    def __init__(self):
        super(prep, self).__init__()

    def scale_r(self, data):
        X = data[0] * self.params[0] + self.params[3]
        y = data[1] * self.params[1] + self.params[4]
        y_x = data[2] * self.params[2] + self.params[5]

        return (X, y, y_x)


    def scale_x(self, x):
        return (x - self.params[3]) / self.params[0]
