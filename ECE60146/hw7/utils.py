import torch
import random
import json
import matplotlib.pyplot as plt
import torchvision.transforms as tvt
from torchvision.utils import make_grid
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3

def show_pizzas(images: torch.Tensor, title: str):
    '''
    Plot pizzas into grid. only plot 4 x 4.
    '''
    grid = make_grid(images[:16, :, :, :], normalize=True, nrow=4)
    plot = tvt.functional.to_pil_image(grid)
    plot.save(f'plot-{title.replace(" ", "-")}.png')

def get_real_pizza_features(dataset, device, samples=1000):
    '''
    Random 1,000 (default) pizzas and extract FID features
    '''
    indices = random.sample(range(len(dataset)), samples)
    pizzas = [dataset[index]['filename'] for index in indices]
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    m, s = fid_score.calculate_activation_statistics(
        pizzas, model, device=device
    )
    
    return m, s

def cal_fid_score(name, m1, s1, genarator, device, letent=100, samples=1000):
    '''
    Calculate FID score generated pizzas
    '''
    # Generate fake pizzas
    noise = torch.randn(samples, letent, 1, 1, device=device)
    fake_pizzas = genarator(noise)
    filenames = []
    import numpy as np
    for i in range(samples):
        grid = make_grid(fake_pizzas[i, :, :, :], normalize=True, nrow=1)
        image = tvt.functional.to_pil_image(grid)
        filename = f'temp/{name}-{i}.jpg'
        image.save(filename)
        filenames.append(filename)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    model = InceptionV3([block_idx]).to(device)
    m2, s2 = fid_score.calculate_activation_statistics(
        filenames, model, batch_size=50, device=device
    )
    return fid_score.calculate_frechet_distance(m1, s1, m2, s2)

def plot_records():
    '''
    Plot grahps to evaluate the results
    '''
    # Load records
    with open(f'BCE-GAN.json') as fp:
        BCE_records = json.load(fp) 
    with open(f'W-GAN.json') as fp:
        W_records = json.load(fp)
    fig, ax = plt.subplots()
    # Plot FID Score
    epoch = [record[0] for record in BCE_records['fid']]
    fid = [record[1] for record in BCE_records['fid']]
    ax.plot(epoch, fid, label='BCE-GAN')
    epoch = [record[0] for record in W_records['fid']]
    fid = [record[1] for record in W_records['fid']]
    ax.plot(epoch, fid, label='W-GAN')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('FID Score')
    ax.set_ylim(0)
    ax.legend(loc='best')
    ax.set_title(f'FID Score')
    fig.savefig(f'result-fid.png')
    # Plot BCE-GAN loss
    fig, ax = plt.subplots()
    ax.plot(BCE_records['epoch'], BCE_records['total'], label='Discrimination')
    ax.plot(BCE_records['epoch'], BCE_records['gen'], label='Generation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    ax.set_title(f'BCE-GAN Training Loss Record')
    fig.savefig(f'result-bce-gan.png')
    # Plot W-GAN loss
    fig, ax = plt.subplots()
    critic_loss = [ # -(loss_real - loss_fake)
       loss[1]-loss[0] for loss in zip(W_records['real'], W_records['fake'])
    ]
    ax.plot(W_records['epoch'], critic_loss, label='Critic')
    ax.plot(W_records['epoch'], W_records['gen'], label='Generation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(loc='best')
    ax.set_title(f'W-GAN Training Loss Record')
    fig.savefig(f'result-W-gan.png')

if __name__ == '__main__':
    plot_records()