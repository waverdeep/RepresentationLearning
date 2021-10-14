import torch.optim as optimizer
import torch.optim.lr_scheduler as scheduler


def get_optimizer(model_parameter, optimizer_param):
    if optimizer_param['optimizer_name'] == 'Adam':
        # We use the Adam optimizer [32] with a learning rate of 2e-4
        return optimizer.Adam(params=model_parameter,
                              lr=optimizer_param['learning_rate'],
                              weight_decay=optimizer_param['weight_decay'],
                              eps=optimizer_param['eps'],
                              amsgrad=optimizer_param['amsgrad'],
                              betas=optimizer_param['betas'])

    elif optimizer_param['optimizer_name'] == 'SGD':
        return optimizer.SGD(params=model_parameter,
                             lr=optimizer_param['learning_rate'],
                             momentum=optimizer_param['momentum'],
                             dampening=optimizer_param['dampening'],
                             weight_decay=optimizer_param['weight_decay'],
                             nesterov=optimizer_param['nesterov'])


def get_scheduler(name, wrapped_optimizer, optimizer_param):
    if name == 'LambdaLR':
        return scheduler.LambdaLR(optimizer=wrapped_optimizer, lr_lambda=optimizer_param['lr_lambda'],
                                  last_epoch=optimizer_param['last_epoch'])




