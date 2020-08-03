class InputExample(object):
  """A single training/test example for token classification."""

  def __init__(self, guid, utterance, context, label):
    """Constructs a InputExample.
    Args:
    guid: Unique id for the example.
    utterance: The utterance of focus for this example.
    context: context of this utterance.
    label: The label for each word of the sequence. 
    specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.utterance = utterance
    self.context = context
    self.label = label