import os, torch
import datetime
from tqdm import tqdm
from accelerate import Accelerator
from .training_module import DiffusionTrainingModule
from .logger import ModelLogger


def launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    args = None,
    val_callback = None,
    val_steps: int = None,
    batch_size: int = 1,
):
    max_steps = None
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        batch_size = getattr(args, 'batch_size', 1)
        max_steps = getattr(args, 'max_steps', None)
    if max_steps is not None:
        # Run enough epochs to exceed max_steps; the inner loop will break early
        num_epochs = max(num_epochs, max_steps)

    optimizer = torch.optim.AdamW(list(model.trainable_modules()), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
    if batch_size > 1:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, batch_size=batch_size,
            collate_fn=lambda x: x, num_workers=num_workers)
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)

    # Allow models to synchronize step counter with logger
    unwrapped_model = model
    if hasattr(unwrapped_model, 'module'):
        unwrapped_model = unwrapped_model.module
    if hasattr(unwrapped_model, 'set_step_counter'):
        unwrapped_model.set_step_counter(model_logger)

    # Ensure all ranks finish slow initialization (dataset scan, wandb.init,
    # model loading, etc.) BEFORE accelerator.prepare() sets up DDP.  Without
    # this barrier, a slow rank-0 (e.g. wandb.init() on a cluster with
    # restricted internet) causes other ranks to timeout inside
    # _verify_param_shape_across_processes during DDP.__init__() in prepare().
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    # Create a GLOO process group for store-based barriers.
    # monitored_barrier only supports GLOO; the default group uses NCCL on
    # GPU training which raises ValueError.  Create once here to reuse across
    # all validation steps.
    _gloo_group = None
    if torch.distributed.is_initialized():
        _gloo_group = torch.distributed.new_group(backend="gloo")

    _reached_max_steps = False
    for epoch_id in range(num_epochs):
        for data in tqdm(dataloader):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if getattr(dataset, 'load_from_cache', False):
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                optimizer.step()
                current_lr = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else learning_rate
                # Collect loss components from model if available
                extra_log_kwargs = {}
                unwrapped = accelerator.unwrap_model(model)
                if hasattr(unwrapped, '_last_loss_components'):
                    extra_log_kwargs = unwrapped._last_loss_components
                    unwrapped._last_loss_components = {}
                model_logger.on_step_end(accelerator, model, save_steps, loss=loss, learning_rate=current_lr, **extra_log_kwargs)
                scheduler.step()
                # Validation callback (rank 0 only; other ranks wait at barrier)
                # NOTE: We use torch.distributed.monitored_barrier instead of
                # accelerator.wait_for_everyone() because the latter uses NCCL
                # ALLREDUCE internally (default 600s timeout).  Validation can
                # easily exceed 600s (e.g. 500+ metric samples + generation),
                # causing NCCL timeout on non-main ranks.  monitored_barrier
                # uses a store-based (TCP/File) barrier that is NOT subject to
                # the NCCL timeout and accepts a configurable wait duration.
                if val_callback and val_steps and model_logger.num_steps % val_steps == 0:
                    if accelerator.is_main_process:
                        val_callback(model_logger.num_steps)
                    if torch.distributed.is_initialized():
                        torch.distributed.monitored_barrier(
                            group=_gloo_group,
                            timeout=datetime.timedelta(minutes=60),
                        )
                    else:
                        accelerator.wait_for_everyone()
                # Early termination for --max_steps
                if max_steps is not None and model_logger.num_steps >= max_steps:
                    _reached_max_steps = True
                    break
        if _reached_max_steps:
            break
        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)
    model_logger.on_training_end(accelerator, model, save_steps)


def launch_data_process_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: ModelLogger,
    num_workers: int = 8,
    args = None,
):
    if args is not None:
        num_workers = args.dataset_num_workers
        
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, collate_fn=lambda x: x[0], num_workers=num_workers)
    model.to(device=accelerator.device)
    model, dataloader = accelerator.prepare(model, dataloader)
    
    for data_id, data in enumerate(tqdm(dataloader)):
        with accelerator.accumulate(model):
            with torch.no_grad():
                folder = os.path.join(model_logger.output_path, str(accelerator.process_index))
                os.makedirs(folder, exist_ok=True)
                save_path = os.path.join(model_logger.output_path, str(accelerator.process_index), f"{data_id}.pth")
                data = model(data)
                torch.save(data, save_path)
