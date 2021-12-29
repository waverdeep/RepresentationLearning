class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new

        return old * self.beta + (1 - self.beta) * new


def update_moving_average(ema_updater, moving_average_model, current_model):
    for current_params, moving_average_params in zip(current_model.parameters(), moving_average_model.parameters()):
        old_weight, update_weight = moving_average_params.data, current_params.data
        moving_average_params.data = ema_updater.update_average(old_weight, update_weight)
