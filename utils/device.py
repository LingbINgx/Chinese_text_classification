

def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}

