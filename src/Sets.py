class Sub:

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def update(self, x, y):
        self.x = x if x is not None else self.x
        self.y = y if y is not None else self.y


class Sets:

    TRAIN = 'training'
    EVAL = 'evaluation'
    TEST = 'test'

    def __init__(self, x_train, y_train, x_eval, y_eval, x_test, y_test):
        self.train = Sub(x_train, y_train)
        self.eval = Sub(x_eval, y_eval)
        self.test = Sub(x_test, y_test)

    def update(self, kind, x, y=None):
        if kind == Sets.TRAIN:
            self.train.update(x, y)
        elif kind == Sets.EVAL:
            self.eval.update(x, y)
        elif kind == Sets.TEST:
            self.test.update(x, y)

    def dup(self, x_train, x_eval, x_test=None):
        x_train = x_train if x_train is not None else self.train.x
        x_eval = x_eval if x_eval is not None else self.eval.x
        x_test = x_test if x_test is not None else self.test.x
        return Sets(x_train, self.train.y, x_eval, self.eval.y, x_test, self.test.y)
