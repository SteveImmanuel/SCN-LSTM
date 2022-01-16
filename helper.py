def print_batch_loss(loss_value: float, current_batch: int, total_batch: int):
    print(' ' * 80, end='\r')
    print(f'Step [{current_batch}/{total_batch}] Loss: {loss_value}', end='\r')


def print_test_overfit(epoch_loss_value: float, current_epoch: int, total_epoch: int):
    print(' ' * 80, end='\r')
    print(f'Epoch [{current_epoch}/{total_epoch}] Loss: {epoch_loss_value}', end='\r')
