import argparse
from solver import Solver

def main(args):
    solver = Solver(root = args.root,
                    result_dir = args.result_dir,
                    img_size = args.img_size,
                    weight_dir = args.weight_dir,
                    batch_size = args.batch_size,
                    gener_batch_size = args.gener_batch_size,
                    g_lr = args.g_lr,
                    d_lr = args.d_lr,
                    beta_1 = args.beta_1,
                    beta_2 = args.beta_2,
                    latent_dim = args.latent_dim,
                    n_critic = args.n_critic,
                    epochs = args.epochs,
                    save_every = args.save_every,
                    diff_aug = args.diff_aug,
                    load_weight = args.load_weight,
                    )
                    
    solver.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='../input/sketch', help='Data location')
    parser.add_argument('--result_dir', type=str, default='test', help='Result images location')
    parser.add_argument('--img_size', type=int, default=32, help='Size of image for discriminator input.')
    parser.add_argument('--weight_dir', type=str, default='weight', help='Weight location')
    parser.add_argument('--batch_size', type=int, default=50, help='Training batch size')
    parser.add_argument('--gener_batch_size', type=int, default=25, help='Batch size for generator.')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='Discriminator Learning rate')
    parser.add_argument('--beta_1', type=float, default=0.0, help='Beta1 for Adam')
    parser.add_argument('--beta_2', type=float, default=0.99, help='Beta2 for Adam')
    parser.add_argument('--save_every', type=int, default=100, help='How often do you want to see the result?')
    parser.add_argument('--latent_dim', type=int, default=128 , help='Latent dimension.')
    parser.add_argument('--n_critic', type=int, default=5, help='n_critic.')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of epoch.')
    parser.add_argument('--diff_aug', type=str, default="translation,cutout,color", help='Data Augmentation')
    parser.add_argument('--load_weight', type=bool, default=False, help='Load weight or not')
                        
    args = parser.parse_args([])
    main(args=args)