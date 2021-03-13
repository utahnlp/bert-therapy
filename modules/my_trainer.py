from transformers import Trainer

SEP_IDX = 3

class MyTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

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
