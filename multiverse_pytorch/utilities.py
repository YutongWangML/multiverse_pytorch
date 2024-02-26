def logits_to_discr(logits):
    last_column = logits[..., -1].unsqueeze(-1)
    return last_column - logits[..., :-1]
