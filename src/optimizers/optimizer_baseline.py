import torch.optim as optimizer
import torch.optim.lr_scheduler as scheduler


def get_optimizer(name, model_parameter, learning_rate, weight_decay, optimizer_param):
    if name == 'Adam':
        # We use the Adam optimizer [32] with a learning rate of 2e-4
        return optimizer.Adam(params=model_parameter, lr=learning_rate, weight_decay=weight_decay,
                              eps=optimizer_param['eps'], amsgrad=optimizer_param['amsgrad'],
                              betas=optimizer_param['betas'])


def get_scheduler(name, wrapped_optimizer, optimizer_param):
    if name == 'LambdaLR':
        return scheduler.LambdaLR(optimizer=wrapped_optimizer, lr_lambda=optimizer_param['lr_lambda'],
                                  last_epoch=optimizer_param['last_epoch'])




