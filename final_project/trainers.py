import torch
from torch import nn
import numpy as np 
import torch.functional as F
from tqdm import tqdm
import os

ALPHA = 1000000

class VAETrainer:
    """
    This class is in charge of train the Variational Autoencoder
    """
    def __init__(self, 
                 model, 
                 optimizer,  
                 train_data_loader,
                 valid_data_loader,
                 epochs,
                 metrics,
                 save_dir='./models/',
                 save_if_improves=True,
                 save_every = None,
                 patience=10,
                 type='vae'):
        
        # device initialization
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = "cpu"
        
        # model initialization
        self.model = model.to(self.device)
        self.type = type
        
        # other params initialization
        self.optimizer = optimizer
        self.metrics = metrics 
        
        # epochs related information
        self.epochs = epochs
        self.start_epoch = 1
        self.checkpoint_dir = save_dir
        
        if save_every is not None and save_if_improves is True:
            print("save_every and save_if_improves are mutually exclusive parameters.\n\
                Check your configuration in the VAETrainer object.")
            exit()
        elif save_every is not None:
            self.save_every = save_every
            self.save_if_improves = False
        else: 
            self.save_every = None
            self.save_if_improves = save_if_improves
        self.patience = patience
        
        # data loader info
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        
        
    def train(self):
        """
        Logic of the whole training
        """
        epochs_w_o_improve = 0
        current_best_loss = np.inf
        epoch_best = self.start_epoch
        for epoch in range(self.start_epoch, self.epochs+1):
            print(f"###################################################################################")
            results_for_epoch = self._train_epoch(epoch)
            train_info, val_info = results_for_epoch
            
            print(f"INFO: Train INFO for epoch {epoch}: ")
            for loss in train_info.keys():
                print(f"{loss}: {train_info[loss]}\n")
                
            print(f"INFO: validation INFO for epoch {epoch}: ")
            for loss in val_info.keys():
                print(f"{loss}: {val_info[loss]}\n")
            
            # update the loss in case it improved
            epoch_loss = val_info['val_loss']
            if epoch_loss < current_best_loss and self.save_if_improves:
                
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                
                print(f"INFO: New best val_loss reached: {val_info['val_loss']} in epoch {epoch}. Saving model...")
                # saving the model
                path_to_save_best_model = os.path.join(self.checkpoint_dir, f"{self.type}_best_{epoch}.pth")
                torch.save(self.model.state_dict(), path_to_save_best_model)
                if epoch > self.start_epoch: # only remove after the model has already a checkpoint
                    os.remove(os.path.join(self.checkpoint_dir, f"{self.type}_best_{epoch_best}.pth"))
                
                # updating values of control
                epoch_best = epoch
                current_best_loss = epoch_loss
                epochs_w_o_improve = 0
            else: 
                epochs_w_o_improve +=1
                
            # if the save_every option is given
            if (self.save_every is not None) and (epoch % self.save_every == 0):
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)
                # saving the model
                torch.save(self.model.state_dict(), path_to_save_best_model)
                
            # if patience limit reached. stop training
            if epochs_w_o_improve == self.patience:
                print(f"WARNING: Patience limit reached. Exiting...\n\
                    Best model saved under {path_to_save_best_model} path.")
                exit()
                
   
    def _train_epoch(self, epoch):
        """
        Logic behind just one epoch of training.
        
        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.
        """
        
        # put model into training mode
        self.model.train()
        
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        
        for (_, data, _) in tqdm(self.train_data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            x = self._reshape(x)
            
            self.optimizer.zero_grad()
            x_recons, mu, logvar, _ = self.model(x)
            loss, loss_recon, loss_kl = self._compute_mse_kl_loss(ALPHA, x, x_recons, mu, logvar)
            # compute gradients and backprop
            loss.backward()
            self.optimizer.step()
            
            # update loss values (reconstructed loss + KL loss)
            total_loss += loss.item()
            total_recon += loss_recon.item()
            total_kl += loss_kl.item() 
            
        # saving it into a dict for INFO loging
        train_log = {
            'loss': total_loss / len(self.train_data_loader),
            'loss_recon': total_recon / len(self.train_data_loader),
            'loss_kl': total_kl / len(self.train_data_loader)
        }
        
        # validation after epoch
        self.model.eval()

        # reset the losses
        total_val_loss = 0.0
        total_val_recon = 0.0
        total_val_kl = 0.0    
        
        with torch.no_grad():
            for batch_idx, (_, data, _) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                x = self._reshape(x)

                x_recons, mu, logvar, _ = self.model(x)
                
                loss, loss_recon, loss_kl = self._compute_mse_kl_loss(ALPHA, x, x_recons, mu, logvar)
                
                # update loss values (reconstructed loss + KL loss)
                total_val_loss += loss.item()
                total_val_recon += loss_recon.item()
                total_val_kl += loss_kl.item() 

        # saving it into a dict for INFO loging
        val_log = {
            'val_loss': total_loss / len(self.valid_data_loader),
            'val_loss_recon': total_recon / len(self.valid_data_loader),
            'val_loss_kl': total_kl / len(self.valid_data_loader)
        }
        
        return train_log, val_log
     
    
    def _compute_mse_kl_loss(self, alpha, x, x_recon, mu, logvar):
        
        # MSE loss
        error = x - x_recon
        recon_loss = torch.mean(torch.square(error), axis=[1,2,3])
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - torch.square(mu) -
                                   torch.exp(logvar), axis=1) 
        
        return torch.mean(alpha*recon_loss + kl_loss, dim=0) , torch.mean(recon_loss), torch.mean(kl_loss)
    
    def _reshape(self, x):
        n_freqBand, n_contextWin = x.size(2), x.size(1)
        return x.view(-1, 1, n_freqBand, n_contextWin) 
    
class ViTTrainer(VAETrainer):
    """
    This class is in charge of train the Visual Transformer
    """
    def __init__(self, 
                 model, 
                 optimizer, 
                 train_data_loader, 
                 valid_data_loader, 
                 epochs, metrics, 
                 save_dir='./models/', 
                 save_if_improves=True, 
                 save_every=None, 
                 patience=10,
                 type='vit'):
        super().__init__(model, 
                         optimizer, 
                         train_data_loader, 
                         valid_data_loader, 
                         epochs, 
                         metrics, 
                         save_dir, 
                         save_if_improves, 
                         save_every, 
                         patience,
                         type)
        
    def train(self):
        super().train()
            
    def _train_epoch(self, epoch):
        # put model into training mode
        self.model.train()
        criterion = nn.BCEWithLogitsLoss()
        
        total_loss = 0.0
        train_acc = 0.0
        for (_, (data, y_true)) in tqdm(self.train_data_loader):
            x = data.type('torch.FloatTensor').to(self.device)
            x = self._reshape(x)
            
            self.optimizer.zero_grad()
            pred_logits = self.model(x).squeeze()
            
            y_pred = torch.round(torch.sigmoid(pred_logits))
            
            loss = criterion(pred_logits, y_true.float())
            train_acc += self._accuracy_fn(y_true, y_pred)
            
            # compute gradients and backprop
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        # saving it into a dict for INFO loging
        train_log = {
            'loss': total_loss / len(self.train_data_loader),
            'accuracy': train_acc / len(self.train_data_loader)
        }
        
        # validation after epoch
        self.model.eval()

        total_val_loss = 0.0 
        val_acc = 0.0  
        with torch.no_grad():
            for batch_idx, (_, (data, y_true)) in enumerate(self.valid_data_loader):
                x = data.type('torch.FloatTensor').to(self.device)
                x = self._reshape(x)

                pred_logits = self.model(x).squeeze()
            
                y_pred = torch.round(torch.sigmoid(pred_logits))
                
                loss = criterion(pred_logits.squeeze(-1), y_true.float())
                val_acc += self._accuracy_fn(y_true, y_pred)
                
                total_val_loss += loss.item()

        # saving it into a dict for INFO loging
        val_log = {
            'val_loss': total_val_loss / len(self.valid_data_loader),
            'val_accuracy': val_acc / len(self.valid_data_loader)
        }
        
        return train_log, val_log
    
    @staticmethod
    def _reshape(x):
        n_freqBand, n_contextWin = x.size(2), x.size(1)
        return x.view(-1, 1, n_freqBand, n_contextWin)  
    
    @staticmethod
    def _accuracy_fn(y_true, y_pred):
        correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
        acc = (correct / len(y_pred))
        return acc
        
        
        
               