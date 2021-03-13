from transformers import Trainer
from transformers.utils import logging
from transformers.optimization import Adafactor, AdamW, get_scheduler
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalPrediction,
    HPSearchBackend,
    PredictionOutput,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    get_last_checkpoint,
    set_seed,
    speed_metrics,
)

SEP_IDX = 3

logger = logging.get_logger(__name__)

class MyTrainer(Trainer):


    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Setup the optimizer and the learning rate scheduler.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through :obj:`optimizers`, or subclass and override this method in a subclass.
        """
        if self.optimizer is None:
            no_decay = ["bias", "LayerNorm.weight"]
            misc_special_embs = ["misc_special_embeddings"]
            if self.args.special_token_lr:
                special_embeddings_named_params = [ (n, p) for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and any(se in n for se in misc_special_embs)]
                logger.info("special_embeddings_params: {}".format([n for (n, p) in special_embeddings_named_params]))
                optimizer_grouped_parameters = [
                    {
                        "params": [p for (n, p) in special_embeddings_named_params],
                        "lr": self.args.special_token_lr,
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay) and not any(se in n for se in misc_special_embs)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls = Adafactor if self.args.adafactor else AdamW
            if self.args.adafactor:
                optimizer_cls = Adafactor
                optimizer_kwargs = {"scale_parameter": False, "relative_step": False}
            else:
                optimizer_cls = AdamW
                optimizer_kwargs = {
                    "betas": (self.args.adam_beta1, self.args.adam_beta2),
                    "eps": self.args.adam_epsilon,
                }
            optimizer_kwargs["lr"] = self.args.learning_rate
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        if self.lr_scheduler is None:
            warmup_steps = (
                self.args.warmup_steps
                if self.args.warmup_steps > 0
                else math.ceil(num_training_steps * self.args.warmup_ratio)
            )

            self.lr_scheduler = get_scheduler(
                self.args.lr_scheduler_type,
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps,
            )


    def compute_loss(self, model, inputs, return_outputs=False):
        """
u       How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        # here, unfold the input into model
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        # Issues: https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/src/transformers/trainer.py#L1468
        # the model outputs is (loss, logits, hidden_states)
        # for logits, we should use outputs[1]
        # https://github.com/huggingface/transformers/blob/9a06b6b11bdfc42eea08fa91d0c737d1863c99e3/src/transformers/models/bert/modeling_bert.py#L1530
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[1]
        logits[labels < SEP_IDX, SEP_IDX:] = logits[labels >= SEP_IDX, :SEP_IDX] = -10000
        p_labels, t_labels = labels < SEP_IDX, labels >= SEP_IDX
        p_logits, t_logits = logits[p_labels, :SEP_IDX], logits[t_labels, SEP_IDX:]
        loss = 0.0
        if sum(p_labels) > 0:
            loss += self.label_smoother({'logits': p_logits}, labels[p_labels])
        if sum(t_labels) > 0:
            loss += self.label_smoother({'logits': t_logits}, labels[t_labels] - SEP_IDX)

        # set label_smoothing_factor to 0 if you want no smoothing
        return (loss, outputs) if return_outputs else loss
