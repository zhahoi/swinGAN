import os
import argparse
import torch
from torchvision.utils import make_grid, save_image

from models import Generator
from utils import gener_noise

def main(args):    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    if args.epochs is not None:
        weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=args.epochs)
    else:
        weight_name = 'checkpoint_1_epoch.pkl'
        
    checkpoint = torch.load(os.path.join(args.weight_dir, weight_name))
    G = Generator().to(device)
    G.load_state_dict(checkpoint['generator_state_dict'])
    G.eval()
    
    if os.path.exists(args.result_dir) is False:
        os.makedirs(args.result_dir)
        
    # For example, img_name = random_55.png
    if args.epochs is None:
        args.epochs = 'latest'
    img_name = 'generated_{epoch}.png'.format(epoch=args.epochs)
    img_path = os.path.join(args.result_dir, img_name)

    # Make latent code and images
    gene_noise = gener_noise(gener_batch_size=args.gener_batch_size, latent_dim=args.latent_dim).to(device)
    generated_imgs = G(gene_noise)

    img_grid = make_grid(generated_imgs, nrow=10, normalize=True, scale_each=True)
    save_image(img_grid, img_path, nrow=10, normalize=True, scale_each=True)  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', type=str, default='./',
                        help='Ouput images location')
    parser.add_argument('--weight_dir', type=str, default='../input/weights',
                        help='Trained weight location of generator. pkl file location')
    parser.add_argument('--img_num', type=int, default=10,
                        help='Generated images number per one input image')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Epoch that you want to see the result. If it is None, the most recent epoch')
    parser.add_argument('--gener_batch_size', type=int, default=100, 
                        help='Batch size for generator.')
    parser.add_argument('--latent_dim', type=int, default=128 , 
                        help='Latent dimension.')

    args = parser.parse_args([])
    main(args)