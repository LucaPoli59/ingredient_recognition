
#
# def _assert_lr_batch_tuning(lr: Optional[float], batch_size: Optional[int]):
#     tune_lr, tune_batch_size = False, False
#     if lr is None:
#         lr, tune_lr = DEF_LR, True
#     if batch_size is None:
#         batch_size, tune_batch_size = DEF_BATCH_SIZE, True
#     return lr, batch_size, tune_lr, tune_batch_size
#
#
# def _tune_batch_size(tuner: Tuner, model: lgn.LightningModule, data_module: ImagesRecipesDataModule,
#                      debug):
#     prev_num_workers = data_module.num_workers
#     data_module.num_workers = 4  # We need to set num_workers to a low number to avoid errors and crashes
#
#     if debug:
#         print("Batch size tuning")
#     tuner.scale_batch_size(model, datamodule=data_module, mode="power", init_val=int(DEF_BATCH_SIZE / 2))
#     if debug:
#         print(f"Batch size tuned: {model.hparams.batch_size}")
#
#     data_module.num_workers = prev_num_workers
#     return model.hparams.batch_size
#
#
# def _tune_lr(tuner: Tuner, model: lgn.LightningModule, data_module: ImagesRecipesDataModule, debug):
#     if debug:
#         print("LR tuning")
#     tuner.lr_find(model, datamodule=data_module, mode="exponential", num_training=10)
#     if debug:
#         print(f"LR tuned: {model.hparams.lr}")
#     return model.hparams.lr