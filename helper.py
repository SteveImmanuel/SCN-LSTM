def print_batch_loss(loss_value: float, current_batch: int, total_batch: int):
    pass


def print_epoch_loss(train_loss_value: float, val_loss_value: float):
    pass


def print_test_overfit(epoch_loss_value: float, current_epoch: int, total_epoch: int):
    print(' ' * 100, end='\r')
    print(f'Epoch [{current_epoch}/{total_epoch}] Loss={epoch_loss_value}', end='\r')
