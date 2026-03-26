import os, glob, re, torch
from accelerate import Accelerator


class ModelLogger:
    def __init__(self, output_path, remove_prefix_in_ckpt=None, state_dict_converter=lambda x:x,
                 wandb_project=None, wandb_run_name=None, wandb_config=None,
                 rolling_save_steps=None, keep_last_k_ckpts=2):
        self.output_path = output_path
        self.remove_prefix_in_ckpt = remove_prefix_in_ckpt
        self.state_dict_converter = state_dict_converter
        self.num_steps = 0
        self.wandb_run = None
        self.rolling_save_steps = rolling_save_steps
        self.keep_last_k_ckpts = keep_last_k_ckpts

        is_main_process = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0))) == 0
        if wandb_project is not None and is_main_process:
            import wandb
            self.wandb_run = wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=wandb_config or {},
            )


    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None, **kwargs):
        self.num_steps += 1

        if self.wandb_run is not None:
            log_dict = {"train/step": self.num_steps}
            if "loss" in kwargs and kwargs["loss"] is not None:
                loss = kwargs["loss"]
                log_dict["train/loss"] = loss.detach().cpu().item() if isinstance(loss, torch.Tensor) else loss
            if "learning_rate" in kwargs and kwargs["learning_rate"] is not None:
                log_dict["train/learning_rate"] = kwargs["learning_rate"]
            # Log extra numeric kwargs (e.g., loss_video, loss_action, video_weight)
            for key, val in kwargs.items():
                if key in ("loss", "learning_rate"):
                    continue
                if isinstance(val, torch.Tensor):
                    log_dict[f"train/{key}"] = val.detach().cpu().item()
                elif isinstance(val, (int, float)):
                    log_dict[f"train/{key}"] = val
            self.wandb_run.log(log_dict, step=self.num_steps)

        # Permanent checkpoints (never deleted)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

        # Rolling checkpoints (high-frequency, only keep last K)
        if self.rolling_save_steps is not None and self.num_steps % self.rolling_save_steps == 0:
            # Skip if this step already saved a permanent checkpoint
            if save_steps is not None and self.num_steps % save_steps == 0:
                return
            self.save_model(accelerator, model, f"rolling-step-{self.num_steps}.safetensors")
            self._cleanup_rolling_checkpoints()


    def on_epoch_end(self, accelerator: Accelerator, model: torch.nn.Module, epoch_id):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, f"epoch-{epoch_id}.safetensors")
            accelerator.save(state_dict, path, safe_serialization=True)


    def on_training_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None):
        if save_steps is not None and self.num_steps % save_steps != 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

        if self.wandb_run is not None:
            self.wandb_run.finish()


    def save_model(self, accelerator: Accelerator, model: torch.nn.Module, file_name):
        accelerator.wait_for_everyone()
        state_dict = accelerator.get_state_dict(model)
        if accelerator.is_main_process:
            state_dict = accelerator.unwrap_model(model).export_trainable_state_dict(state_dict, remove_prefix=self.remove_prefix_in_ckpt)
            state_dict = self.state_dict_converter(state_dict)
            os.makedirs(self.output_path, exist_ok=True)
            path = os.path.join(self.output_path, file_name)
            accelerator.save(state_dict, path, safe_serialization=True)

    def _cleanup_rolling_checkpoints(self):
        pattern = os.path.join(self.output_path, "rolling-step-*.safetensors")
        ckpts = sorted(glob.glob(pattern), key=lambda p: int(re.search(r"rolling-step-(\d+)", p).group(1)))
        for old in ckpts[:-self.keep_last_k_ckpts]:
            os.remove(old)
