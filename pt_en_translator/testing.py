import os
import random
import training
import config
import models
from checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import pickle


def create_trainer(checkpoint_path):

    max_epochs = config.get_max_epochs()
    acc_batch = config.get_accumulate_batches()

    checkpoint_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(checkpoint_dir, '{epoch}-{step:.2f}'),
                                          save_top_k=-1)  # Keeps all checkpoints.
    resume_from_checkpoint = None
    if os.path.exists(checkpoint_path):
        print(f'Restoring checkpoint: {checkpoint_path}')
        resume_from_checkpoint = checkpoint_path

    trainer = pl.Trainer(gpus=1,
                         max_epochs=max_epochs,
                         check_val_every_n_epoch=1,
                         profiler=True,
                         accumulate_grad_batches=acc_batch,
                         checkpoint_callback=checkpoint_callback,
                         callbacks = [checkpoint_callback],
                         progress_bar_refresh_rate=50,
                         resume_from_checkpoint=resume_from_checkpoint)
    #                     distributed_backend='dp')

    return trainer


def test(x_test, checkpoint_path):
    model = models.create_models(x_test, x_test, x_test)
    trainer = training.create_trainer(checkpoint_path)
    trainer.test(model)


if __name__ == "__main__":
    checkpoint_path = "save/0.pkl"

    with open("test_gt.pkl", "rb") as f:
        test_pkl = pickle.load(f)

    test(test_pkl, checkpoint_path)
