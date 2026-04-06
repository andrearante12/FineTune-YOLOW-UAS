import torch
ckpt = torch.load('5_epoch.pt', map_location='cpu', weights_only=False)
print('epoch:', ckpt.get('epoch'))
print('best_fitness:', ckpt.get('best_fitness'))
print('train_metrics:', ckpt.get('train_metrics'))
print('train_results keys:', list(ckpt.get('train_results', {}).keys()) if isinstance(ckpt.get('train_results'),
dict) else ckpt.get('train_results'))
