import torch

################################################################
###     Functions                                            ###
################################################################
def get_device():
    # If CUDA is available print devices
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    return device


MODEL_STATE_DICT = 'model_state_dict'
OPTIMIZER_STATE_DICT = 'optimizer_state_dict'
EPOCH_STATE = 'epoch'
TRAIN_LOSS_HISTORY_STATE = 'train_loss_history'
VAL_LOSS_HISTORY_STATE = 'val_loss_history'
BEST_VAL_LOSS_STATE = 'best_val_loss'

def save_model_state(filename, model, optimizer, epoch, train_loss_history, val_loss_history, best_val_loss):
    torch.save({
        MODEL_STATE_DICT: model.state_dict(),
        OPTIMIZER_STATE_DICT: optimizer.state_dict(),
        EPOCH_STATE: epoch,
        TRAIN_LOSS_HISTORY_STATE: train_loss_history,
        VAL_LOSS_HISTORY_STATE: val_loss_history,
        BEST_VAL_LOSS_STATE: best_val_loss
    }, filename)

def load_model_state(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[MODEL_STATE_DICT])
    optimizer.load_state_dict(checkpoint[OPTIMIZER_STATE_DICT])
    last_epoch = checkpoint[EPOCH_STATE]
    train_loss_history = checkpoint[TRAIN_LOSS_HISTORY_STATE]
    val_loss_history = checkpoint[VAL_LOSS_HISTORY_STATE]
    best_val_loss = checkpoint[BEST_VAL_LOSS_STATE]

    return model, optimizer, last_epoch, train_loss_history, val_loss_history, best_val_loss

def build_model_state():
    last_epoch = 0
    train_loss_history = []
    val_loss_history = []
    best_val_loss = None

    return last_epoch, train_loss_history, val_loss_history, best_val_loss