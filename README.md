# Again-vcSkip to content
Search or jump to…
Pulls
Issues
Codespaces
Marketplace
Explore
 
@Najafi2022 
Pegahshafiei
/
againvc-new
Public
Fork your own copy of Pegahshafiei/againvc-new
Code
Issues
Pull requests
Projects
Security
Insights
Pegahshafiei/againvc-new
Latest commit
@Pegahshafiei
Pegahshafiei Update README.md
…
36 minutes ago
Git stats
 56
Files
Type
Name
Latest commit message
Commit time
agent
inorm on content embedding
2 years ago
config
rm __init__.py
2 years ago
data
preprocess
2 years ago
dataloader
remove neg, pos
2 years ago
indexer
train
2 years ago
model
inorm on content embedding
2 years ago
preprocessor
train
2 years ago
util
np2pt: enforce to use FloatTensor
2 years ago
.gitignore
inference
2 years ago
LICENSE
Create LICENSE
2 years ago
README.md
Update README.md
36 minutes ago
inference.ipynb
Update inference.ipynb
2 years ago
inference.py
add model.png
2 years ago
make_indexes.py
update config
2 years ago
model.png
add model.png
2 years ago
preprocess.py
update config
2 years ago
train.py
add model.png
2 years ago
README.md
AGAIN-VC
This is the official implementation of the paper AGAIN-VC: A One-shot Voice Conversion using Activation Guidance and Adaptive Instance Normalization. AGAIN-VC is an auto-encoder-based model, comprising of a single encoder and a decoder. With a proper activation as an information bottleneck on content embeddings, the trade-off between the synthesis quality and thespeaker similarity of the converted speech is improved drastically.

The demo page is here, and the pretrained model is available here.

The figure shows the model overview. The left part is the encoder, while the right part is the decoder. Note that L1 Loss is to make the input mel-spectrogram and the output as close as possible.



Usage
Preprocessing
python preprocess.py [--config <CONFIG>] [--njobs <NJOBS>]

# Example:
python preprocess.py -c config/preprocess.yaml
Preprocessing the wave files into acoustic features (eg. mel-spectrogram). Note that we provide a tiny subset of VCTK corpus in this repo just for checking whether the code works or not. If you want to use the whole VCTK corpus, please make sure to revise the preprocessing config file first.

Making indexes for training
python make_indexes.py [--config <CONFIG>]

# Example
python make_indexes.py -c config/make_indexes.yaml
Splitting the train/dev set from the preprocessed features.

Training
python train.py 
                [--config <CONFIG>]
                [--dry] [--debug] [--seed <SEED>]
                [--load <LOAD>]
                [--njobs <NJOBS>] 
                [--total-steps <TOTAL_STEPS>]
                [--verbose-steps <VERBOSE_STEPS>] 
                [--log-steps <LOG_STEPS>]
                [--save-steps <SAVE_STEPS>]
                [--eval-steps <EVAL_STEPS>]
                
# Example
python train.py \
  -c config/train_again-c4s.yaml \
  --seed 1234567 \
  --total-steps 100000
Note we use wandb as the default training logger. You can also use other training logger like tensorboard, but you need to edit util/mylogger.py first.

Inference
python inference.py
                    --load <LOAD>
                    --source <SOURCE>
                    --target <TARGET>
                    --output <OUTPUT>
                    [--config <CONFIG>]
                    [--dsp-config <DSP_CONFIG>]
                    [--seglen <SEGLEN>] [--dry] [--debug] [--seed <SEED>]
                    [--njobs <NJOBS>]

# Example
python inference.py \
  -c config/train_again-c4s.yaml \
  -l checkpoints/again/c4s \
  -s data/wav48/p225/p225_001.wav \
  -t data/wav48/p226/p226_001.wav \
  -o data/generated
Colab
We also provide a google colab for inference: https://colab.research.google.com/drive/1Q3v2bTKPV0jB1F_dBT1YuqINDq9a52qO?usp=sharing

1.target

Using an encoder to separate speaker and content information to improve synthesis quality and reduce model size at the same time.
Introducing the activation guide, applying an additional activation function as an information bottleneck to guide content embeddings in continuous space, to improve the trade-off between synthesis quality and separation ability.
2.Describe innovation

The use of encoder and decoder blocks or the use of convolutional and recurrent neural networks

3.change

import torch import torch.nn as nn import torch.nn.functional as F import torch.optim.lr_scheduler as lrSched import os from util.mytorch import np2pt

def build_model(build_config, device, mode): model = Model(**build_config.model.params).to(device) if mode == 'train': # model_state, step_fn, save, load optimizer = torch.optim.Adam(model.parameters(), **build_config.optimizer.params) scheduler = lrSched.ExponentialLR(optimizer,gamma=0.99994) criterion_l1 = nn.L1Loss() criterion_l2 = nn.CrossEntropyLoss() model_state = { 'model': model, 'optimizer': optimizer, 'scheduler':scheduler, 'steps': 0, # static, no need to be saved 'criterion_l1': criterion_l1, 'criterion_l2':criterion_l2, 'device': device, 'grad_norm': build_config.optimizer.grad_norm, # this list restores the dynamic states '_dynamic_state': [ 'model', 'optimizer', 'steps' ] } return model_state, train_step elif mode == 'inference': model_state = { 'model': model, # static, no need to be saved 'device': device, '_dynamic_state': [ 'model' ] } return model_state, inference_step else: raise NotImplementedError

For training and evaluating
def train_step(model_state, data, train=True): meta = {} model = model_state['model'] optimizer = model_state['optimizer'] scheduler = model_state['scheduler'] criterion_l1 = model_state['criterion_l1'] criterion_l2 = model_state['criterion_l2'] device = model_state['device'] grad_norm = model_state['grad_norm']

if train:
    optimizer.zero_grad()
    model.train()
else:
    model.eval()

x = data['mel'].to(device)
label = data['label'].to(device)
#data speaker label should be retrieved
dec, speaker = model(x)
loss_rec = criterion_l1(dec, x)
loss_speaker = criterion_l2(speaker,label)
loss = loss_rec + loss_speaker

if train:
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),
        max_norm=grad_norm)
    optimizer.step()
    # scheduler.step()

with torch.no_grad():
    model.eval()
    m_src = x[0][None]
    m_tgt = x[1][None]
    c_src = x[0][None]
    s_tgt = x[1][None]
    s_src = x[0][None]

    dec = model.inference(c_src, s_tgt)
    rec = model.inference(c_src, s_src)


meta['log'] = {
    'loss_rec': loss_rec.item(),
    'loss_speaker': loss_speaker.item(),
    'total_loss': loss.item()
}
meta['mels'] = {
    'src': m_src,
    'tgt': m_tgt,
    'dec': dec,
    'rec': rec,
}

return meta
def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

For inference
def inference_step(model_state, data): meta = {} model = model_state['model'] device = model_state['device'] model.to(device) model.eval()

source = data['source']['mel']
target = data['target']['mel']

source = np2pt(source).to(device)
target = np2pt(target).to(device)

dec = model.inference(source, target)
meta = {
    'dec': dec
}
return meta
====================================
Modules
====================================
class InstanceNorm(nn.Module): def init(self, eps=1e-5): super().init() self.eps = eps

def calc_mean_std(self, x, mask=None):
    B, C = x.shape[:2]

    mn = x.view(B, C, -1).mean(-1)
    sd = (x.view(B, C, -1).var(-1) + self.eps).sqrt()
    mn = mn.view(B, C, *((len(x.shape) - 2) * [1]))
    sd = sd.view(B, C, *((len(x.shape) - 2) * [1]))
    
    return mn, sd


def forward(self, x, return_mean_std=False):
    mean, std = self.calc_mean_std(x)
    x = (x - mean) / std
    if return_mean_std:
        return x, mean, std
    else:
        return x
class ConvNorm(torch.nn.Module): def init(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, groups=1, bias=True, w_init_gain='linear', padding_mode='zeros', sn=False): super(ConvNorm, self).init() if padding is None: assert(kernel_size % 2 == 1) padding = int(dilation * (kernel_size - 1) / 2)

    self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups,
                                bias=bias, padding_mode=padding_mode)
    if sn:
        self.conv = nn.utils.spectral_norm(self.conv)

    torch.nn.init.xavier_uniform_(
        self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal
class ConvNorm2d(torch.nn.Module): def init(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, dilation=1, bias=True, w_init_gain='linear', padding_mode='zeros'): super().init() if padding is None: if type(kernel_size) is tuple: padding = [] for k in kernel_size: assert(k % 2 == 1) p = int(dilation * (k - 1) / 2) padding.append(p) padding = tuple(padding) else: assert(kernel_size % 2 == 1) padding = int(dilation * (kernel_size - 1) / 2)

    self.conv = torch.nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, dilation=dilation,
                                bias=bias, padding_mode=padding_mode)

    torch.nn.init.xavier_uniform_(
        self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

def forward(self, signal):
    conv_signal = self.conv(signal)
    return conv_signal
class EncConvBlock(nn.Module): def init(self, c_in, c_h, subsample=1): super().init() self.seq = nn.Sequential( ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3, stride=subsample), ) self.subsample = subsample def forward(self, x): y = self.seq(x) if self.subsample > 1: x = F.avg_pool1d(x, kernel_size=self.subsample) return x + y

class DecConvBlock(nn.Module): def init(self, c_in, c_h, c_out, upsample=1): super().init() self.dec_block = nn.Sequential( ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3), ) self.gen_block = nn.Sequential( ConvNorm(c_in, c_h, kernel_size=3, stride=1), nn.BatchNorm1d(c_h), nn.LeakyReLU(), ConvNorm(c_h, c_in, kernel_size=3), ) self.upsample = upsample def forward(self, x): y = self.dec_block(x) if self.upsample > 1: x = F.interpolate(x, scale_factor=self.upsample) y = F.interpolate(y, scale_factor=self.upsample) y = y + self.gen_block(y) return x + y

====================================
Model
====================================
class Encoder(nn.Module): def init( self, c_in, c_out, n_conv_blocks, c_h, subsample ): super().init() # self.conv2d_blocks = nn.ModuleList([ # ConvNorm2d(1, 8), # ConvNorm2d(8, 8), # ConvNorm2d(8, 1), # nn.BatchNorm2d(1), nn.LeakyReLU(), # ]) # 1d Conv blocks self.inorm = InstanceNorm() self.conv1d_first = ConvNorm(c_in * 1, c_h) self.conv1d_blocks = nn.ModuleList([ EncConvBlock(c_h, c_h, subsample=sub) for _, sub in zip(range(n_conv_blocks), subsample) ]) self.out_layer = ConvNorm(c_h, c_out)

def forward(self, x):

    y = x
    # for block in self.conv2d_blocks:
    #     y = block(y)
    y = y.squeeze(1)
    y = self.conv1d_first(y)

    mns = []
    sds = []
    
    for block in self.conv1d_blocks:
        y = block(y)
        y, mn, sd = self.inorm(y, return_mean_std=True)
        mns.append(mn)
        sds.append(sd)

    y = self.out_layer(y)

    return y, mns, sds
class Decoder(nn.Module): def init( self, c_in, c_h, c_out, n_conv_blocks, upsample ): super().init() self.in_layer = ConvNorm(c_in, c_h, kernel_size=3) self.act = nn.LeakyReLU()

    self.conv_blocks = nn.ModuleList([
        DecConvBlock(c_h, c_h, c_h, upsample=up) for _, up in zip(range(n_conv_blocks), upsample)
    ])

    self.inorm = InstanceNorm()
    self.rnn = nn.GRU(c_h, c_h, 2)
    self.out_layer = nn.Linear(c_h, c_out)

def forward(self, enc, cond, return_c=False, return_s=False):
    y1, _, _ = enc
    y2, mns, sds = cond
    mn, sd = self.inorm.calc_mean_std(y2)
    c = self.inorm(y1)
    c_affine = c * sd + mn

    y = self.in_layer(c_affine)
    y = self.act(y)

    for i, (block, mn, sd) in enumerate(zip(self.conv_blocks, mns, sds)):
        y = block(y)
        y = self.inorm(y)
        y = y * sd + mn

    y = torch.cat((mn, y), dim=2)
    y = y.transpose(1,2)
    y, _ = self.rnn(y)
    y = y[:,1:,:]
    y = self.out_layer(y)
    y = y.transpose(1,2)
    if return_c:
        return y, c
    elif return_s:
        mn = torch.cat(mns, -2)
        sd = torch.cat(sds, -2)
        s = mn * sd
        return y, s
    else:
        return y
class VariantSigmoid(nn.Module): def init(self, alpha): super().init() self.alpha = alpha def forward(self, x): y = 1 / (1+torch.exp(-self.alpha*x)) return y

class NoneAct(nn.Module): def init(self): super().init() def forward(self, x): return x

class Activation(nn.Module): dct = { 'none': NoneAct, 'sigmoid': VariantSigmoid, 'tanh': nn.Tanh } def init(self, act, params=None): super().init() self.act = Activation.dctact

def forward(self, x):
    return self.act(x)
class SpeakerRecognition(nn.Module): def init(self): super().init() self.listofspeakers = os.listdir("./data/wav48") self.dense_layers = nn.Sequential( nn.Linear(3072, 128), nn.LeakyReLU(0.2), nn.Linear(128, len(self.listofspeakers)) ) # self.softmax = nn.Softmax(dim=1)

def forward(self,input_data):
    logits = self.dense_layers(input_data)
    # predictions = self.softmax(logits)
    # return predictions
    return logits
class Model(nn.Module): def init(self, encoder_params, decoder_params, activation_params): super().init()

    self.encoder = Encoder(**encoder_params)
    self.decoder = Decoder(**decoder_params)
    self.speakerRecognition = SpeakerRecognition()
    self.act = Activation(**activation_params)

def forward(self, x, x_cond=None):

    len_x = x.size(2)
    
    if x_cond is None:
        x_cond = torch.cat((x[:,:,len_x//2:], x[:,:,:len_x//2]), axis=2)

    x, x_cond = x[:,None,:,:], x_cond[:,None,:,:]

    enc, mns_enc, sds_enc = self.encoder(x) # , mask=x_mask)
    cond, mns_cond, sds_cond = self.encoder(x_cond) #, mask=cond_mask)

    enc = (self.act(enc), mns_enc, sds_enc)
    cond = (self.act(cond), mns_cond, sds_cond)
    


    y = self.decoder(enc, cond)
    yenc, ymns_enc, ysds_enc = self.encoder(y)

    inputlayer=ymns_enc+ysds_enc
    mnsdvector = torch.cat(inputlayer,dim=1)
    shape = mnsdvector.shape
    mnsdvector = mnsdvector.reshape(shape[0],shape[1])
    speaker = self.speakerRecognition(mnsdvector)
    return y,speaker

def inference(self, source, target):

    original_source_len = source.size(-1)
    original_target_len = target.size(-1)

    if original_source_len % 8 != 0:
        source = F.pad(source, (0, 8 - original_source_len % 8), mode='reflect')
    if original_target_len % 8 != 0:
        target = F.pad(target, (0, 8 - original_target_len % 8), mode='reflect')

    x, x_cond = source, target

    # x_energy = x.mean(1, keepdims=True).detach()
    # x_energy = x_energy - x_energy.min()
    # x_mask = (x_energy > x_energy.mean(-1, keepdims=True)/2).detach()

    # cond_energy = x_cond.mean(1, keepdims=True).detach()
    # cond_energy = cond_energy - cond_energy.min()
    # cond_mask = (cond_energy > cond_energy.mean(-1, keepdims=True)/2).detach()

    x = x[:,None,:,:]
    x_cond = x_cond[:,None,:,:]

    enc, mns_enc, sds_enc = self.encoder(x) # , mask=x_mask)
    cond, mns_cond, sds_cond = self.encoder(x_cond) #, mask=cond_mask)

    enc = (self.act(enc), mns_enc, sds_enc)
    cond = (self.act(cond), mns_cond, sds_cond)

    y = self.decoder(enc, cond)

    dec = y[:,:,:original_source_len]

    return dec
5.https://github.com/KimythAnly/AGAIN-VC

6.Pegah Shafiei, senior student of medical engineering at South Tehran University with student number 40014140111043, dsp course

7.https://drive.google.com/file/d/1sTpefreBEOJa9DKSO5jI6VhxGbzZDd1s/view?usp=drivesdk

8.https://aparat.com/v/PSy9q https://aparat.com/v/oj5zq

9.pro

About
No description, website, or topics provided.
Resources
 Readme
License
 MIT license
Stars
 0 stars
Watchers
 1 watching
Forks
 0 forks
Releases
No releases published
Packages
No packages published
Contributors 2
@KimythAnly
KimythAnly
@Pegahshafiei
Pegahshafiei
Languages
Python
95.9%
 
Jupyter Notebook
4.1%
Footer
© 2023 GitHub, Inc.
Footer navigation
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
Pegahshafiei/againvc-new
