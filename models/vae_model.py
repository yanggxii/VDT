import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
  def __init__(self, cfg):
      super(VAE, self).__init__()
      
      self.dr = 0.25
      self.ln1 = nn.Linear(32*2, 32)

      ############# encoder
      self.module_encoder = nn.Sequential(
                                        nn.Linear(768, 512),
                                        nn.LayerNorm(512),
                                        nn.ReLU(), 
                                        nn.Dropout(p=self.dr),
                                        nn.Linear(512, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(), 
                                        nn.Dropout(p=self.dr), 
                                        nn.Linear(256, 128),                     
                                      )
      self.fc_mu = nn.Linear(128, 128)   # 128
      self.fc_var = nn.Linear(128, 128)
      

      ############# decoder
      self.module_decoder = nn.Sequential(
                                        nn.Linear(128, 128),
                                        nn.LayerNorm(128),
                                        nn.ReLU(), 
                                        nn.Dropout(p=self.dr),                                          
                                        nn.Linear(128, 256),
                                        nn.LayerNorm(256),
                                        nn.ReLU(), 
                                        nn.Dropout(p=self.dr),
                                        nn.Linear(256, 512),
                                        nn.LayerNorm(512),
                                        nn.ReLU(), 
                                        nn.Dropout(p=self.dr),
                                        nn.Linear(512, 768),                              
                                      )

  def vae_loss_function(self, x, y, mu, log_var, uncertainty_weights=None, mode=None):
      num_iter = 0
      num_iter += 1
      loss_type = 'H'
      kld_weight = 1  # Account for the minibatch samples from the dataset
      beta = 1.5
      gamma = 1000.
      max_capacity = 25
      Capacity_max_iter = 1e5
      C_max = torch.Tensor([max_capacity])
      C_stop_iter = Capacity_max_iter
      
      batch_size = x.size()[0]
          
      recons_loss =F.mse_loss(x, y)

      kld_loss = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1)
      
      if mode == 'test':
          kld_loss = (1 + uncertainty_weights) * kld_loss
          kld_loss = torch.mean(kld_loss, dim = 0)
      else:
          kld_loss = torch.mean(kld_loss, dim = 0)
      if loss_type == 'H': # https://openreview.net/forum?id=Sy2fzU9gl
          loss = recons_loss + beta * kld_weight * kld_loss
      elif loss_type == 'B': # https://arxiv.org/pdf/1804.03599.pdf
          C_max = C_max.to(x.device)
          C = torch.clamp(C_max/C_stop_iter * num_iter, 0, C_max.data[0])
          loss = recons_loss + gamma * kld_weight* (kld_loss - C).abs()
      else:
          raise ValueError('Undefined loss type.')
          
      return loss


  def decoder(self, z):
      y = self.module_decoder(z)
      return y

  def sampling(self, mu, log_var):
      std = torch.exp(0.5 * log_var)
      eps = torch.randn_like(std)
      z = eps.mul(std).add_(mu)
      return z

  def encoder(self, x):
      h = self.module_encoder(x)
      mu = self.fc_mu(h)
      log_var = self.fc_var(h)

      return mu, log_var

  def forward(self, x1=None, x2=None, mode='train'):
    if mode == 'train':
      mu_src, log_var_src = self.encoder(x1)
      mu_tgt, log_var_tgt = self.encoder(x2)
      
      z_src = self.sampling(mu_src, log_var_src)
      z_tgt = self.sampling(mu_tgt, log_var_tgt)

      y_src = self.decoder(z_src)
      y_tgt = self.decoder(z_tgt)

      
      loss_src = self.vae_loss_function(y_src, x1, mu_src, log_var_src)
      loss_tgt = self.vae_loss_function(y_tgt, x2, mu_tgt, log_var_tgt)

    
      out_dict = {'F_s':{'z':  (z_src), 'mu_s': mu_src, 'var_s': log_var_src, 'loss': loss_src},
                  'F_t':{'z': (z_tgt), 'mu_t': mu_tgt, 'var_t': log_var_tgt, 'loss': loss_tgt}}
      
      return out_dict
    
    if mode == 'test':
      mu_tgt, log_var_tgt = self.encoder(x1)
      z_tgt = self.sampling(mu_tgt, log_var_tgt)
      y_tgt = self.decoder(z_tgt)
      loss_tgt = self.vae_loss_function(y_tgt, x1, mu_tgt, log_var_tgt)
            
      out_dict = {'F_t':{'z': (mu_tgt), 'mu_t': mu_tgt, 'var_t': log_var_tgt, 'y_t':y_tgt}}
      
      return out_dict