import torch
import torch.optim as optim
from torchvision.utils import make_grid, save_image

from tensorboardX import SummaryWriter
import time
import datetime
from tqdm import tqdm

from utils import *
from models import *
from dataloader import *
from diff_aug import * 

# for reproductionary
init_torch_seeds(seed=1234)

class Solver():
    def __init__(self, root='dataset/anime_faces', result_dir='result', img_size=32, weight_dir='weight', load_weight=False,
                 batch_size=32, gener_batch_size=25, epochs=200, save_every=100, latent_dim=128, n_critic=5, diff_aug=None, 
                 g_lr=0.0002, d_lr=0.0001, beta_1=0.0, beta_2=0.99, logdir=None):
        
        # cpu or gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load generator and discriminator
        self.G = Generator()
        self.G.to(self.device)
        self.D = Discriminator(diff_aug=diff_aug)
        self.D.to(self.device)

        # load training dataset
        self.train_loader, _ = data_loader(root=root, batch_size=batch_size, shuffle=True, 
                                                img_size=img_size, mode='train')

        # optimizer
        self.optim_D = optim.Adam(self.D.parameters(), lr=d_lr, betas=(beta_1, beta_2))
        self.optim_G = optim.Adam(self.G.parameters(), lr=g_lr, betas=(beta_1, beta_2))

        # Some hyperparameters
        self.latent_dim = latent_dim
        self.writer = SummaryWriter(logdir)

        # Extra things
        self.result_dir = result_dir
        self.weight_dir = weight_dir
        self.load_weight = load_weight
        self.epochs = epochs
        self.gener_batch_size = gener_batch_size
        self.img_size = img_size
        self.start_epoch = 0
        self.num_epoch = epochs
        self.save_every = save_every
        self.n_critic = n_critic
        
    '''
    <show_model >
    Print model architectures
    '''
    def show_model(self):
        print('================================ Discriminator for image =====================================')
        print(self.D)
        print('==========================================================================================\n\n')
        print('================================= Generator ==================================================')
        print(self.G)
        print('==========================================================================================\n\n')
        
    '''
        < set_train_phase >
        Set training phase
    '''
    def set_train_phase(self):
        self.D.train()
        self.G.train()
    
    '''
        < load_checkpoint >
        If you want to continue to train, load pretrained weight from checkpoint
    '''
    def load_checkpoint(self, checkpoint):
        print('Load model')
        self.D.load_state_dict(checkpoint['discriminator_image_state_dict'])
        self.G.load_state_dict(checkpoint['generator_state_dict'])
        self.optim_D.load_state_dict(checkpoint['optim_d'])
        self.optim_G.load_state_dict(checkpoint['optim_g'])
        self.start_epoch = checkpoint['epoch']
        
    '''
        < save_checkpoint >
        Save checkpoint
    '''
    def save_checkpoint(self, state, file_name):
        print('saving check_point')
        torch.save(state, file_name)
    
    '''
        < all_zero_grad >
        Set all optimizers' grad to zero 
    '''
    def all_zero_grad(self):
        self.optim_D.zero_grad()
        self.optim_G.zero_grad()

    '''
        < train >
        Train the D_image, D_latnet, G and E 
    '''
    def train(self):
        if self.load_weight is True:
            weight_name = 'checkpoint_{epoch}_epoch.pkl'.format(epoch=self.epochs)
            checkpoint = torch.load(os.path.join(self.weight_dir, weight_name))
            self.load_checkpoint(checkpoint)
        
        self.set_train_phase()
        self.show_model()


        print('====================     Training    Start... =====================')
        for epoch in range(self.start_epoch, self.num_epoch):
            start_time = time.time()

            for iters, img in tqdm(enumerate(self.train_loader)):
                # load real images
                real_imgs = img.type(torch.cuda.FloatTensor)

                # generate fake images
                noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], self.latent_dim)))
                fake_imgs = self.G(noise)

                ''' ----------------------------- 1. Train D ----------------------------- '''
                # discriminator predict 
                fake_validity = self.D(fake_imgs.detach())
                real_validity = self.D(real_imgs)

                # gradient_penalty
                gradient_penalty = calculate_gradient_penalty(self.D, real_imgs, fake_imgs.detach())
                # loss measures generator's ability to fool the discriminator
                err_d = wgan_loss(real_validity, real_or_not=True) + wgan_loss(fake_validity, real_or_not=False) 

                d_loss = gradient_penalty + err_d

                # update D
                self.all_zero_grad()
                d_loss.backward()
                self.optim_D.step()

                self.writer.add_scalars('losses', {'d_loss': d_loss, 'grad_penalty': gradient_penalty}, iters)

                ''' ----------------------------- 2. Train G ----------------------------- '''
                # train the generator every n_critic iterations
                if iters % self.n_critic == 0:
                    noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (img.shape[0], self.latent_dim)))
                    generated_imgs = self.G(noise)
                    
                    fake_validity = self.D(generated_imgs)
                    g_loss = wgan_loss(fake_validity, real_or_not=True).to(self.device)

                    # update G
                    self.all_zero_grad()
                    g_loss.backward()
                    self.optim_G.step()

                    self.writer.add_scalars('losses', {'g_loss': g_loss}, iters)

                log_file = open('log.txt', 'w')
                log_file.write(str(epoch))

                # Print error, save intermediate result image and weight
                if epoch and iters % self.save_every == 0:
                    et = time.time() - start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    print('[Elapsed : %s /Epoch : %d / Iters : %d] => loss_d : %f / loss_g : %f / gradient_penalty : %f '\
                          %(et, epoch, iters, d_loss.item(), g_loss.item(), gradient_penalty.item()))

                    # Save intermediate result image
                    if os.path.exists(self.result_dir) is False:
                        os.makedirs(self.result_dir)

                    # Generate fake image
                    self.G.eval()
                    with torch.no_grad():
                        gene_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (self.gener_batch_size, self.latent_dim)))
                        generated_imgs= self.G(gene_noise)
                        sample_imgs = generated_imgs[:25]

                    img_name = 'generated_img_{epoch}_{iters}.jpg'.format(epoch=epoch, iters=(iters % len(self.train_loader)))
                    img_path = os.path.join(self.result_dir, img_name)

                    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)
                    save_image(img_grid, img_path, nrow=5, normalize=True, scale_each=True)  

                    # Save intermediate weight
                    if os.path.exists(self.weight_dir) is False:
                        os.makedirs(self.weight_dir)  

            # Save weight at the end of every epoch
            if (epoch % 5) == 0:
                # self.save_weight(epoch=epoch)
                checkpoint = {
                    "generator_state_dict": self.G.state_dict(),
                    "discriminator_image_state_dict": self.D.state_dict(),
                    "optim_g": self.optim_G.state_dict(),
                    "optim_d": self.optim_D.state_dict(),
                    "epoch": epoch
                    }
                path_checkpoint = os.path.join(self.weight_dir, "checkpoint_{}_epoch.pkl".format(epoch))
                self.save_checkpoint(checkpoint, path_checkpoint)        