from transformers import Trainer

SEP_IDX = 3

class MyTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs[0]
        logits[labels < SEP_IDX, SEP_IDX:] = logits[labels >= SEP_IDX, :SEP_IDX] = -10000
        p_labels, t_labels = labels < SEP_IDX, labels >= SEP_IDX
        p_logits, t_logits = logits[p_labels, :SEP_IDX], logits[t_labels, SEP_IDX:]
        # set label_smoothing_factor to 0 if you want no smoothing
        loss = self.label_smoother({'logits': p_logits}, labels[p_labels]) + self.label_smoother({'logits': t_logits}, labels[t_labels] - SEP_IDX)
        return (loss, outputs) if return_outputs else loss
