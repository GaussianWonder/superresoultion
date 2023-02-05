import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 0.2989, 0.5870, 0.1140 in terms of perceived visibility
RGB_WEIGHTS = torch.FloatTensor([65.481, 128.553, 24.966]).to(DEVICE)

IMAGENET_MEAN = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
IMAGENET_STD = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)

IMAGENET_MEAN_CUDA = torch.FloatTensor([0.485, 0.456, 0.406]).to(DEVICE).unsqueeze(0).unsqueeze(2).unsqueeze(3)
IMAGENET_STD_CUDA = torch.FloatTensor([0.229, 0.224, 0.225]).to(DEVICE).unsqueeze(0).unsqueeze(2).unsqueeze(3)
