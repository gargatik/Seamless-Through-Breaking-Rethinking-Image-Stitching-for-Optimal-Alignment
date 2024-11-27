from .base_options import BaseOptions
from yacs.config import CfgNode as CN

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--which_epoch', type=str, default='16', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=2000, help='how many test images to run')
        self.isTrain = False

        return parser
    
    # Namespace(aspect_ratio=1.0, batchSize=32, checkpoints_dir='./checkpoints', de_root='/project/liutaorong/dataset/DPED100K/image/', fineSize=256, gpu_ids=[0], how_many=2000, init_gain=0.02, init_type='normal', input_mask_root='/project/liutaorong/dataset/DPED100K/mask/', input_nc=6, isTrain=False, lambda_L1=1, lambda_P=0.04, lambda_S=250, model='training1', nThreads=2, n_layers_D=3, name='TransRef', ndf=64, ngf=64, norm='instance', ntest=inf, num_workers=8, output_nc=3, phase='test', ref_root='/project/liutaorong/dataset/DPED100K/reference/', results_dir='./results/', use_dropout=False, which_epoch='16')
    def get_test_option_cfg(self):
        option_cfg = CN()
        option_cfg.aspect_ratio = 1.0
        option_cfg.batchSize = 32
        option_cfg.checkpoints_dir = './checkpoints'
        option_cfg.de_root = '/project/liutaorong/dataset/DPED100K/image/'
        option_cfg.fineSize = 256
        option_cfg.gpu_ids = [0]
        option_cfg.how_many = 2000
        option_cfg.init_gain = 0.02
        option_cfg.init_type = 'normal'
        option_cfg.input_mask_root = '/project/liutaorong/dataset/DPED100K/mask/'
        option_cfg.input_nc = 6
        option_cfg.isTrain = False
        option_cfg.lambda_L1 = 1
        option_cfg.lambda_P = 0.04
        option_cfg.lambda_S = 250
        option_cfg.model = 'training1'
        option_cfg.nThreads = 2
        option_cfg.n_layers_D = 3
        option_cfg.name = 'TransRef'
        option_cfg.ndf = 64
        option_cfg.ngf = 64
        option_cfg.norm = 'instance'
        option_cfg.ntest = float("inf")
        option_cfg.num_workers = 8
        option_cfg.output_nc = 3
        option_cfg.phase = 'test'
        option_cfg.ref_root = '/project/liutaorong/dataset/DPED100K/reference/'
        option_cfg.results_dir = './results/'
        option_cfg.use_dropout = False
        option_cfg.which_epoch = 16
        return option_cfg
    
        