    # input_model = copy.deepcopy(model)
    # last_model = copy.deepcopy(model)
    # last_model.load_weights_from_checkpoint(os.path.join(self._save_dir, "checkpoints", "last.ckpt"))
    # best_model = copy.deepcopy(model)
    #
    # ckpt_c = self.checkpoint_callback
    # best_model_path = ckpt_c.best_model_path if ckpt_c is not None else None
    # # IF the path exist and is not the last one
    # if (self._debug or best_model_path is None or best_model_path == ""
    #         or not os.path.exists(best_model_path) or ckpt_c.best_model_score is None
    #         or ckpt_c.best_model_score.item() == ckpt_c.current_score.item()):
    #     # if the best model path is not correct we take the last one
    #     best_model_path = os.path.join(self._save_dir, "checkpoints", "last.ckpt")
    #
    # best_model.load_weights_from_checkpoint(best_model_path)
    # os.rename(best_model_path, os.path.join(self._save_dir, "best_model.ckpt"))
    # if os.path.exists(os.path.join(self._save_dir, "checkpoints", "last.ckpt")):
    #     os.remove(os.path.join(self._save_dir, "checkpoints", "last.ckpt"))
    # return input_model, last_model, best_model
    #
    #
    #
    #
    # m = copy.deepcopy(lgn_model)
    #
    # input_m, last_m, best_m = trainer.fit(
    #     model=lgn_model,
    #     datamodule=data_module,
    #     ckpt_path=ckpt_path
    # )
    #
    # if debug:
    #     print("Training completed")
    #
    # print("Predicting with the before train model")
    # trainer.test(model=lgn_model, datamodule=data_module)  # TODO: rimuovere
    #
    # print("Predicting with the before train model copied")
    # trainer.test(model=m, datamodule=data_module)  # TODO: rimuovere
    #
    # print("Predicting with the init model")
    # trainer.test(model=input_m, datamodule=data_module)  # TODO: rimuovere
    #
    # print("Predicting with the last model")
    # trainer.test(model=last_m, datamodule=data_module)  # TODO: rimuovere
    #
    # print("Predicting with the best model")
    # trainer.test(model=best_m, datamodule=data_module)