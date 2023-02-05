import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

rgb_weights = torch.FloatTensor([65.481, 128.553, 24.966]).to(DEVICE)
imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(DEVICE).unsqueeze(0).unsqueeze(2).unsqueeze(3)
imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(DEVICE).unsqueeze(0).unsqueeze(2).unsqueeze(3)

