import torch

from trainer import Trainer
from loss import loss_DANN
from models import DANNModel
from dataloader.data_loader import create_data_generators
from metrics import AccuracyScoreFromLogits
from utils.callbacks import simple_callback, ModelSaver
import configs.dann_config as dann_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    print("Creating datasets")
    train_gen_s, val_gen_s, test_gen_s = create_data_generators(dann_config.DATASET,
                                                                dann_config.SOURCE_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)

    train_gen_t, val_gen_t, test_gen_t = create_data_generators(dann_config.DATASET,
                                                                dann_config.TARGET_DOMAIN,
                                                                batch_size=dann_config.BATCH_SIZE,
                                                                infinite_train=True,
                                                                image_size=dann_config.IMAGE_SIZE,
                                                                num_workers=dann_config.NUM_WORKERS,
                                                                device=device)
    print("Creating model")
    model = DANNModel().to(device)
    acc = AccuracyScoreFromLogits()

    tr = Trainer(model, loss_DANN)
    print("Starting training")
    tr.fit(train_gen_s, train_gen_t,
           n_epochs=1,
           validation_data=[val_gen_s, val_gen_t],
           metrics=[acc],
           steps_per_epoch=1,
           callbacks=[simple_callback, ModelSaver("DANN")])
