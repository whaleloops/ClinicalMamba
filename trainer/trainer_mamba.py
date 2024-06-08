from transformers import Trainer
from transformers.trainer_utils import is_main_process
from transformers.utils import logging
import torch
import os
import math

logger = logging.get_logger(__name__)

def get_custom_scheduler(optimizer, num_warmup_steps, num_training_steps, min_lr=1e-2):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        # Transition from warmup to cosine decay
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed = (1 - min_lr) * cosine_decay + min_lr
        return decayed

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class MambaTrainer(Trainer):
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     input_ids = inputs.pop("input_ids")
    #     lm_logits = model(input_ids).logits
    #     labels = input_ids.to(lm_logits.device)
    #     shift_logits = lm_logits[:, :-1, :].contiguous()
    #     labels = labels[:, 1:].contiguous()
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     lm_loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
    #     return lm_loss

    # def create_scheduler(self, num_training_steps: int, optimizer: torch.optim.Optimizer = None):
    #     """
    #     Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
    #     passed as an argument.

    #     Args:
    #         num_training_steps (int): The number of training steps to do.
    #     """
    #     if self.lr_scheduler is None:
    #         # self.lr_scheduler = get_scheduler(
    #         #     self.args.lr_scheduler_type,
    #         #     optimizer=self.optimizer if optimizer is None else optimizer,
    #         #     num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
    #         #     num_training_steps=num_training_steps,
    #         #     scheduler_specific_kwargs=self.args.lr_scheduler_kwargs,
    #         # )
    #         assert self.args.learning_rate > 1e-5
    #         self.lr_scheduler = get_custom_scheduler(
    #             self.optimizer if optimizer is None else optimizer, 
    #             self.args.get_warmup_steps(num_training_steps), 
    #             num_training_steps, 
    #             min_lr=1e-5/self.args.learning_rate
    #         )
    #         self._created_lr_scheduler = True
    #     return self.lr_scheduler
    
    def save_model(self, output_dir=None, _internal_call=False):
        if output_dir is None:
            output_dir = self.args.output_dir 
        if is_main_process(self.args.local_rank):
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
            logger.info(f"Saving model checkpoint to {output_dir}")
                
            torch.save(self.model.state_dict(), f"{output_dir}/pytorch_model.bin")
            self.tokenizer.save_pretrained(output_dir)
        