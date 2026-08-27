import torch
torch.set_float32_matmul_precision('high')


def base_training(trainer, dm, lit_mod, ckpt=None,only_rec=False):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    if ckpt:
        print(f"loading from ckpt {ckpt}")
        # map_location: load a GPU-trained checkpoint on a CPU-only machine
        # (and vice-versa) without a device mismatch error.
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lit_mod.load_state_dict(
            torch.load(ckpt, weights_only=True, map_location=device)['state_dict']
        )

    #trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)
    #trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt)
    if only_rec:
        print("ONLY RECONSTRUCTION")
        print(ckpt)
        trainer.test(lit_mod, datamodule=dm, ckpt_path=ckpt)
    else:
        trainer.fit(lit_mod, datamodule=dm)
        trainer.test(lit_mod, datamodule=dm, ckpt_path='best')

def multi_dm_training(trainer, dm, lit_mod, test_dm=None, test_fn=None, ckpt=None):
    if trainer.logger is not None:
        print()
        print("Logdir:", trainer.logger.log_dir)
        print()

    trainer.fit(lit_mod, datamodule=dm, ckpt_path=ckpt)

    if test_fn is not None:
        if test_dm is None:
            test_dm = dm
        lit_mod._norm_stats = test_dm.norm_stats()

        best_ckpt_path = trainer.checkpoint_callback.best_model_path
        trainer.callbacks = []
        trainer.test(lit_mod, datamodule=test_dm, ckpt_path=best_ckpt_path)

        print("\nBest ckpt score:")
        print(test_fn(lit_mod).to_markdown())
        print("\n###############")
