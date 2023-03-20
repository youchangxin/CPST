import itertools
import torch
from .base_model import BaseModel
from util.image_pool import ImagePool
from . import networks
from . import cpst_net
from . import edgeDetection
import torch.nn as nn
from torch.nn import init
from torchvision.transforms import RandomVerticalFlip
from .networks import calc_mean_std, mean_variance_norm, get_key


class CPSTModel(BaseModel):
    """ This class implements CPST model.
    This code is inspired by DCLGAN
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CPST """
        parser.add_argument('--CPST_mode', type=str, default="CPST", choices='CPST')
        parser.add_argument('--disable_wavelet', action="store_true", help="to disable the Wavelet transform module")
        parser.add_argument('--shallow_layer', action="store_true",
                            help="if specified, also use features of shallow layers")
        parser.add_argument('--skip_connection_3', action='store_true',
                            help='if specified, add skip connection on ReLU-3')
        parser.add_argument('--lambda_GAN_Adversarial', type=float, default=1.5,
                            help='weight for GAN loss：GAN(G(Ic, Is))')
        parser.add_argument('--lambda_GAN_D', type=float, default=0.2, help='weight for GAN loss：GAN(G(Is, Ic))')
        parser.add_argument('--lambda_style', type=float, default=0.1, help='weight for global style loss')
        parser.add_argument('--lambda_local', type=float, default=0.1, help='weight for local style loss')
        parser.add_argument('--lambda_content', type=float, default=1.0, help='weight for global content loss')
        parser.add_argument('--lambda_GAN_Line', type=float, default=0.0, help='weight for Line loss')
        parser.add_argument('--lambda_CYC', type=float, default=0.0,
                            help='weight for l1 reconstructe loss:||Ic - G(G(Ic, Is),Ic)||')

        opt, _ = parser.parse_known_args()

        # Set default parameters for CPST.
        if opt.CPST_mode.lower() == "cpst":
            pass
        else:
            raise ValueError(opt.CPST_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G']
        self.visual_names = ['real_A', 'real_B', 'fake_B']

        if self.opt.lambda_GAN_Adversarial > 0.0 and self.isTrain:
            self.loss_names += ['adversarial']

        if self.opt.lambda_GAN_Line > 0.0 and self.isTrain:
            self.loss_names += ['line']

        if self.opt.lambda_CYC > 0.0 and self.isTrain:
            self.visual_names += ['rec_B']
            self.loss_names += ['cyc']

        if self.opt.lambda_style > 0.0 and self.isTrain:
            self.loss_names += ['style']

        if self.opt.lambda_local > 0.0 and self.isTrain:
            self.loss_names += ['local']

        if self.opt.lambda_content > 0.0 and self.isTrain:
            self.loss_names += ['content']

        if self.opt.lambda_GAN_D > 0.0 and self.isTrain:
            self.loss_names += ['D']

        if self.isTrain:
            self.model_names = ["netTransformer", "netDec", "netD"]
        else:  # during test time, only load G
            self.model_names = ['netTransformer', "netDec"]

        # define networks
        self.netEnc = cpst_net.Encoder(opt.disable_wavelet)
        self.netTransformer = cpst_net.Transformer(shallow_layer=opt.shallow_layer)
        self.netDec = cpst_net.Decoder(skip_connection_3=opt.skip_connection_3, disable_wavelet=True)
        init_net(self.netTransformer, 'normal', 0.02, self.gpu_ids)
        init_net(self.netDec, 'normal', 0.02, self.gpu_ids)
        self.netEnc.to(self.device)
        self.netTransformer.to(self.device)
        self.netDec.to(self.device)

        if self.isTrain:
            # load edge detection model
            if self.opt.lambda_GAN_Line > 0.0:
                self.detection = edgeDetection.DoobNet()
                self.detection.load_state_dict(torch.load('models/doobnet.pth.tar', map_location="cpu")['state_dict'])
                self.detection.to(self.device)
                self.criterionLine = nn.BCELoss().to(self.device)

            self.fake_pool = ImagePool(opt.pool_size)
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D,
                                          opt.crop_size, opt.feature_dim, opt.max_conv_dim,
                                          opt.normD, opt.init_type, opt.init_gain, opt.no_antialias,
                                          self.gpu_ids, opt)

            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionCyc = torch.nn.L1Loss().to(self.device)
            self.criterionMSE = torch.nn.MSELoss().to(self.device)

            # define optimizer
            self.optimizer_G = torch.optim.Adam(
                itertools.chain(self.netTransformer.parameters(), self.netDec.parameters()),
                lr=opt.lr_G, betas=(opt.beta1, opt.beta2)
            )
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def optimize_parameters(self):
        # forward
        self.forward()
        # update D
        if self.opt.lambda_GAN_D:
            self.set_requires_grad([self.netD], True)
            self.set_requires_grad([self.netTransformer, self.netDec], False)
            self.optimizer_D.zero_grad()
            self.loss_D = self.backward_D()
            self.loss_D.backward(retain_graph=True)
            self.optimizer_D.step()

        # update G
        self.set_requires_grad([self.netD], False)
        self.set_requires_grad([self.netTransformer, self.netDec], True)
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.s_feats, self.c_feats, h_feats, s_pool_feats = self.netEnc(self.real_A, self.real_B)
        cs_feat, adain_feat_3, pool_adain_feats = self.netTransformer(self.c_feats, self.s_feats, h_feats, s_pool_feats)
        self.fake_B = self.netDec(cs_feat, pool_adain_feats, adain_feat_3)

        if self.opt.lambda_CYC > 0.0 and self.isTrain:
            rec_s_feats, rec_c_feats, rec_h_feats = self.netEnc(self.real_B, self.fake_B)
            rec_cs_feat, rec_adain_feat_3 = self.netTransformer(rec_c_feats, rec_s_feats, rec_h_feats)
            self.rec_B = self.netDec(rec_cs_feat, rec_adain_feat_3)

    def backward_D_basic(self, netD, style, fake):
        """Calculate GAN loss for the discriminator
        Parameters:
            netD (network)       -- the discriminator D
            style (tensor array) -- real style images
            fake (tensor array)  -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = netD(style)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        return loss_D

    def backward_D(self):
        """Calculate GAN loss for discriminator D"""
        if self.opt.lambda_GAN_D > 0.0:
            fake_B = self.fake_pool.query(self.fake_B)
            self.loss_D = self.backward_D_basic(self.netD, self.real_B, fake_B) * self.opt.lambda_GAN_D
        else:
            self.loss_D = 0

        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and loss for the generator"""
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN_Adversarial > 0.0:
            pred_fakeB = self.netD(self.fake_B)
            self.loss_adversarial = self.criterionGAN(pred_fakeB, True).mean() * self.opt.lambda_GAN_Adversarial
        else:
            self.loss_adversarial = 0.0

        # Line Loss
        if self.opt.lambda_GAN_Line > 0.0:
            edges = torch.cat([self.real_A, self.fake_B], dim=0)
            self.detection.eval()
            with torch.no_grad():
                edges = self.detection(edges)
                y, y_ = torch.chunk(edges, 2, dim=0)
            self.loss_line = self.opt.lambda_GAN_Line * self.criterionLine(y_, y)
        else:
            self.loss_line = 0.0

        # L1 Cycle Loss
        if self.opt.lambda_CYC > 0.0:
            self.loss_cyc = self.criterionCyc(self.rec_B, self.real_B) * self.opt.lambda_CYC
        else:
            self.loss_cyc = 0

        # global style loss
        if self.opt.lambda_style > 0.0:
            self.loss_style = torch.tensor(0., device=self.device)
            stylized_feats, _ = self.netEnc.styEnc(self.fake_B)
            for i in range(1, 5):
                s_feats_mean, s_feats_std = calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = calc_mean_std(stylized_feats[i])
                self.loss_style += self.criterionMSE(stylized_feats_mean, s_feats_mean) +\
                                   self.criterionMSE(stylized_feats_std, s_feats_std)
            self.loss_style = self.opt.lambda_style * self.loss_style
        else:
            self.loss_style = 0

        # local style loss
        if self.opt.lambda_local > 0.0:
            self.loss_local = torch.tensor(0., device=self.device)
            for i in range(2, 5):
                c_key = get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i],
                                                     std * networks.mean_variance_norm(self.c_feats[i]) + mean)
            self.loss_local = self.opt.lambda_local * self.loss_local

        else:
            self.loss_local = 0

        # content loss
        if self.opt.lambda_content > 0.0:
            self.loss_content = torch.tensor(0., device=self.device)
            stylized_feats, _ = self.netEnc.conEnc(self.fake_B)
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(networks.mean_variance_norm(stylized_feats[i]),
                                                       networks.mean_variance_norm(self.c_feats[i]))
            self.loss_content = self.opt.lambda_style * self.loss_content
        else:
            self.loss_content = 0

        self.loss_G = self.loss_cyc + self.loss_adversarial + self.loss_line \
                      + self.loss_style + self.loss_content + self.loss_local

        return self.loss_G


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net
