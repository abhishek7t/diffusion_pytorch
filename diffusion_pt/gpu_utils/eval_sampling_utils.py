import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torch import nn
from torchvision import models


# We will extract the features from the pool_3 layer and logits
class InceptionFeatureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inception_v3 = models.inception_v3(pretrained=True, aux_logits=False)
        self.inception_v3.eval() # set the model to evaluation mode
        self.pool3 = nn.AdaptiveAvgPool2d((1, 1)) # Equivalent to ppol_3 in tf
        
    def forward(self, x):
        # N x 3 x 299 x 299
        x = self.inception_v3.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.inception_v3.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.inception_v3.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.inception_v3.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.inception_v3.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.inception_v3.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.inception_v3.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.inception_v3.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.inception_v3.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.inception_v3.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.inception_v3.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.inception_v3.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.inception_v3.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.inception_v3.avgpool(x)
        # N x 2048 x 1 x 1
        x = self.inception_v3.dropout(x)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 2048
        logits = self.inception_v3.fc(x)
        
        return {'pool_3': x, 'logits': logits}
        

class InceptionFeatures:
    """
    Compute and store Inception features for a dataset
    """
    
    def __init__(self, dataset, device, limit_dataset_size=0, batch_size=32, num_workers=4):
        if limit_dataset_size > 0:
            dataset = Subset(dataset, list(range(limit_dataset_size)))
        
        # Create DataLoader for the dataset
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        self.device = device
        self.model = InceptionFeatureExtractor().to(self.device)
        self.model.eval() # set the model to evaluation mode
        
        self.cached_inception_real = None # cached inception features
        self.real_inception_score = None # saved inception scores for the dataset
        
    def get(self):
        # On the first invocation, compute Inception activations for the eval dataset
        if self.cached_inception_real is None:
            print('computing inception features on the eval set...')
            inception_real_batches = {'pool_3': [], 'logits': []}
            tstart = time.time()
            
            with torch.no_grad():
                for batch in self.dataloader:
                    images = batch['images'].to(self.device)
                    output = self.model(images)
                    
                    # collect features
                    inception_real_batches['pool_3'].append(output['pool_3'].cpu().numpy())
                    inception_real_batches['logits'].append(output['logits'].cpu().numpy())

            # concatenate batches of features and convert to numpy
            self.cached_inception_real = {
                feat_key: np.concatenate(inception_real_batches[feat_key], axis=0).astype(np.float64)
                for feat_key in ['pool_3', 'logits']
            }
            
            print('Cached eval inception tensors: logits: {}, pool_3: {} (time: {.2f}s)'.format(
                self.cached_inception_real['logits'].shape,
                self.cached_inception_real['pool_3'].shape,
                time.time() - tstart
            ))
            
            # compute the Inception Score (based on logits)
            self.real_inception_score = self._calculate_inception_score(
                self.cached_inception_real['logits']
            )
            del self.cached_inception_real['logits'] # Save memory by deleting logits after computing score
        
        print('Real inception score: ', self.real_inception_score)
        return self.cached_inception_real, self.real_inception_score
    
    def _calculate_inception_score(self, logits):
        """
        Compute the Inception score from logits (dummy implementation here for simplicity).
        You would use a real classifier metric like Frechet Inception Distance (FID) or Inception Score (IS).
        """
        return None
