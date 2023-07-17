import torch
import torchvision.transforms as tvt
from scipy.stats import wasserstein_distance
from PIL import Image
from pathlib import Path

folder = Path('.') / 'hw2' # set folder location

def find_w_dist(image_a, image_b):
    '''
    This function takes two images and calculate Wasserstein distance
    '''
    num_bins = 10
    distance = []
    # compose transformation
    transfrom = tvt.Compose([
        tvt.ToTensor(), 
        tvt.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    # transfrom both images
    image_a = transfrom(image_a)
    image_b = transfrom(image_b)
    # find Wasserstein distance one channel at a time
    for ch in range(3):
        hist_a = torch.histc(image_a[ch], bins=num_bins, min=-1, max=-1)
        hist_a = hist_a.div(hist_a.sum())
        hist_b = torch.histc(image_b[ch], bins=num_bins, min=-1, max=-1)
        hist_b = hist_b.div(hist_b.sum())
        
        distance.append(wasserstein_distance(
            hist_a.numpy(),
            hist_b.numpy()
        ))
    # print out the sumation of Wasserstein distance from all 3 channels
    print(f'Sum of Wasserstein distance is: {sum(distance):.3f}')

stop_sign_1 = Image.open(folder / 'stop_sign1.jpg')
stop_sign_2 = Image.open(folder / 'stop_sign2.jpg')
# Wasserstein distance before tranformation
find_w_dist(stop_sign_1, stop_sign_2)

# perform perspective transformation
persective_image = tvt.functional.perspective(
    stop_sign_2,
    [[185, 164], [317, 238], [317, 349], [186, 296]],
    [[44, 169], [458, 167], [464, 335], [41, 338]]
)
# Wasserstein distance after perspective tranformation
find_w_dist(stop_sign_1, persective_image)

# Test with affine transformation
affine_transfomer = tvt.RandomAffine(degrees=(0, 180), translate=(0.1, 0.3))
for _ in range(10):
    affime_image = affine_transfomer(stop_sign_2)
    find_w_dist(stop_sign_1, affime_image)
