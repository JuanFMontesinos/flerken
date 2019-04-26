#!/usr/bin/env python3
# -*- coding: utf-8 -*-
__author__ = "Juan Montesinos"
__version__ = "0.2.3"
__maintainer__ = "Juan Montesinos"
__email__ = "juanfelipe.montesinos@upf.edu"
__status__ = "Prototype"
import torch
import os
import uuid
import numpy as np
import logging
import sys
import time
import shutil
import datetime
from collections import OrderedDict
import subprocess
from . import utils as ptutils,network_initialization,sqlite_tools
from .traceback_gradients import tracegrad
from tqdm import tqdm
from tensorboardX import SummaryWriter
from functools import partial
LOGGING_FORMAT_B = "[%(filename)s: %(funcName)s] %(message)s]"
LOGGIN_FORMAT_A = "[%(asctime)-15s %(filename)s:%(lineno)d %(funcName)s] %(message)s]"

def set_training(func):
    def inner(*args,**kwargs):
        self = args[0]
        self.hyperparameters()
        self.__atribute_assertion__()
        if not hasattr(self, 'init_function') :
        	self.init_function = partial(network_initialization.init_weights,init_type = self.initilizer)
        self._train()
        self.key['LR'] = self.LR
        self.key['OPTIMIZER'] = str(self.optimizer)
        self.key['EPOCH'] = 0
        self.key['MODEL'] = self.model_version
        self.key['ITERATIONS'] = 0
        self.__update_db__()
        return func(*args,**kwargs)
    return inner
def checkpoint_on_key(func):
    def inner(*args,**kwargs):
        self = args[0]
        self.key['CHECKPOINT'] = 1
        self.__update_db__()
        return func(*args,**kwargs)
    return inner     
def dl():
    return torch.rand(2),[torch.rand(5),torch.rand(5)]
def create_folder(path):
   if not os.path.exists(path):
       os.umask(0) #To mask the permission restrictions on new files/directories being create
       os.makedirs(path,0o755) # setting permissions for the folder
class framework(object):
    def __init__(self,model,rootdir,workname, *args,**kwargs):
        self.model = model
        self.model_version = ''
        self.rootdir = rootdir
        self.workname = workname
        self.RESERVED_KEYS = ['DATE_OF_CREATION','MODEL','LR','LOSS','ACC','ID']
        if not os.path.isdir(self.rootdir):
            raise Exception('Rootdir set is not a directory')
        self.db_name = 'experiment_db.sqlite'
        self.db_dir = os.path.join(self.rootdir,self.db_name)
        self.db = sqlite_tools.sq(self.db_dir)
        self.key = {}
        if self.workname is not None:
            self.resume = self.db.exists(self.workname)
            if self.resume:
                self.workdir = os.path.join(self.rootdir,self.workname)
            else:
                if not os.path.exists(self.workname):
                    raise Exception('Workname should point to pretrained weights or to be None')
        else:
            self.workdir = None
            self.resume = False
    def __update_db__(self):
        
        if type(self.key) == dict:
            self.db.update(self.key) 
        else: 
            raise Exception('Trying to update a Null key dictionary')
    def __repr__(self):
        return 'PyTorch framework created {0} at {1}'.format(self.key['DATE_OF_CREATION'],self.rootdir)
    def __call__(self):
        self.print_key()
    def __setloggers__(self,**kwargs):
        ptutils.setup_logger('train_iter_log',os.path.join(self.workdir,'train_iter_log.txt'),**kwargs)
        self.train_iter_logger = logging.getLogger('train_iter_log')

        ptutils.setup_logger('val_epoch_log',os.path.join(self.workdir,'val_epoch_log.txt'),**kwargs)
        self.val_epoch_logger = logging.getLogger('val_epoch_log')        

        ptutils.setup_logger('error_log',os.path.join(self.workdir,'err.txt'),**kwargs)
        self.err_logger = logging.getLogger('error_log')
    def __setup_experiment__(self,**kwargs):
        self.start_epoch = 0
        self.absolute_iter = 0
        now = datetime.datetime.now()
        self.workname = str(uuid.uuid4())[:7]
        self.workdir =os.path.join(self.rootdir,self.workname)    
        create_folder(self.workdir)
        ptutils.setup_logger('model_logger',os.path.join(self.workdir,'model_architecture.txt') ,writemode='w')
        self.model_logger  = logging.getLogger('model_logger')
        self.model_logger.info('Model Version: {0}'.format(self.model_version))
        self.model_logger.info(self.model)
        ### TRAIN ITERATION LOGGER ###
        self.__setloggers__(writemode='w',to_console=False)

        self.key = {'ID':self.workname,'MODEL':self.model_version,'DATE_OF_CREATION':now.strftime("%Y-%m-%d %H:%M")}
        self.db.insert_value(self.key)
    def print_info(self,log):

        result = subprocess.Popen(["nvidia-smi", "--format=csv", "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        nvidia = result.stdout.readlines().copy()
        nvidia = [str(x) for x in nvidia]
        nvidia = [x[2:-3]+'\r\t' for x in nvidia]
        acum = ''
        for x in nvidia:
            acum = acum + x
        

        log.info('\r\t __Python VERSION: {0} \r\t'
                 '__pyTorch VERSION: {1} \r\t'
                 '__CUDA VERSION: {2}\r\t'
                 '__CUDNN VERSION: {3} \r\t'
                 '__Number CUDA Devices: {4} \r\t'
                 '__Devices: {5}'
                 'Active CUDA Device: GPU {6} \r\t'
                 'Available devices {7} \r\t'
                 'Current cuda device {8} \r\t'.format(sys.version,torch.__version__,torch.version.cuda,torch.backends.cudnn.version(),torch.cuda.device_count(),
                                      acum,torch.cuda.current_device(), torch.cuda.device_count(),torch.cuda.current_device()))

class pytorchfw(framework):
    def __init__(self,model,rootdir,workname,main_device,trackgrad):
        super(pytorchfw,self).__init__(model,rootdir,workname)
        self.checkpoint_name = 'checkpoint.pth'  
        self.cuda = torch.cuda.is_available()
        self.main_device = 'cuda:{0}'.format(int(main_device)) if self.cuda and main_device != 'cpu' else 'cpu'
        self.main_device = torch.device(self.main_device)
        if main_device != 'cpu':
            self.model.to(self.main_device)
        self.inizilizable_layers = [self.model.parameters()]
        self.trackgrad = trackgrad
        self.assertion_variables = ['initilizer','EPOCHS','optimizer','criterion','LR','dataparallel']
        if trackgrad:
            self.tracker=tracegrad(self.model,verbose=False)
    def _allocate_tensor(self,x,device = None):
        if device is None:
            device = self.main_device
        else:
            device = torch.device(device)
        if self.dataparallel:
            return x
        else:
            if isinstance(x,list):
                return [i.to(device) for i in x]
            elif isinstance(x,tuple):
                return tuple([i.to(device) for i in x])
            else:
                return x.to(device)
            
    def _loadcheckpoint(self):
        directory = os.path.join(self.workdir,'best'+self.checkpoint_name)
        ### TRAIN ITER LOGGER ###
        self.__setloggers__(writemode='a',to_console=False)
        ptutils.setup_logger('train_iter_log', os.path.join(self.workdir, 'train_iter_log.txt'), writemode='a',to_console=False)
        self.train_iter_logger= logging.getLogger('train_iter_log')
        ### VAL EPOCH LOGGER ###
        ptutils.setup_logger('val_epoch_log', os.path.join(self.workdir, 'val_epoch_log.txt'),  writemode='a',to_console=False)
        self.val_epoch_logger = logging.getLogger('val_epoch_log')

        if os.path.isfile(directory):
            print("=> Loading checkpoint '{}'".format(directory))
            checkpoint = torch.load(directory, map_location=lambda storage, loc: storage)
            self.start_epoch = checkpoint['epoch']
            self._load_model(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.batch_data= checkpoint['loss']
            self.absolute_iter = checkpoint['iter']
            self.key = checkpoint['key']
            print("=> Loaded checkpoint '{}' (epoch {})"
                  .format(directory, checkpoint['epoch']))
        else: 
            print("=> No checkpoint found at '{}'".format(directory))
    @checkpoint_on_key
    def save_checkpoint(self, filename=None):
        state = {
            'epoch': self.epoch + 1,
            'iter' : self.absolute_iter + 1,
            'arch': self.model_version,
            'state_dict': self.model.state_dict(),
            'optimizer' : self.optimizer.state_dict(),
            'loss': self.batch_data,
            'key': self.key
            }
        if filename is None:
            filename = os.path.join(self.workdir,self.checkpoint_name)
            
        elif isinstance(filename,str):
            filename = os.path.join(self.workdir,filename)
        print('Saving checkpoint at : {}'.format(filename))
        torch.save(state, filename)
        if self.batch_data.is_best():
            shutil.copyfile(filename, os.path.join(self.workdir,'best'+self.checkpoint_name))  
        print('Checkpoint saved sucessfully')
    def _load_model(self,directory,strict_loading=True,from_checkpoint=False,**kwargs):
        print('Loading pretrained weights')
        if isinstance(directory,dict):
            state_dict = directory
        else:
            state_dict = torch.load(directory, map_location=lambda storage, loc: storage)
        if 'checkpoint' in directory:
            state_dict = state_dict['state_dict']
        new_state_dict = OrderedDict()
        
        for k, v in state_dict.items():
            if k[:7] == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict,strict=strict_loading)        
    def train_epoch(self,logger):
        j = 0
        self.train_iterations =  len(iter(self.train_loader))
        with tqdm(self.train_loader,desc='Epoch: [{0}/{1}]'.format(self.epoch,self.EPOCHS)) as pbar:
            for gt,inputs,visualization in pbar :
                try:
                    self.absolute_iter += 1
                    
                    inputs = self._allocate_tensor(inputs)
                    
                    self.batch_data.update_timed()
                    output = self.model(*inputs) if isinstance(inputs,list) else self.model(inputs)
                    
                    try:
                        device = torch.device(self.outputdevice)
                    except:
                        if isinstance(output,(list,tuple)):
                            device = output[0].device
                        else:
                            device = output.device
                    
                    gt = self._allocate_tensor(gt,device=device)    
                    loss = self.criterion(output, gt) 
                    # compute gradient and do SGD step
                    self.optimizer.zero_grad()
                    loss.backward()
    
                    self.gradients()
                    if self.trackgrad:
                        self.tracker(self.model.named_parameters())
                    self.optimizer.step()
                    self.batch_data.loss = loss.data.item()
                    self.tensorboard_writer(loss.data.item(),output,gt,self.absolute_iter,visualization)
                    self.batch_data.update_loss()
                    self.batch_data.update_timeb()
                    self.batch_data.end = time.time()
                    pbar.set_postfix(loss=self.batch_data.loss)
    
                    self.batch_data.print_logger(self.epoch,j,self.train_iterations,logger)
                    j+=1

                except Exception as e:
                    try:
                        self.save_checkpoint(filename = os.path.join(self.workdir,'checkpoint_backup.pth'))
                    except:
                        self.err_logger.error('Failed to deal with exception. Couldnt save backup at {0} \n'
                                              .format(os.path.join(self.workdir,'checkpoint_backup.pth')))
                    self.err_logger.error(str(e))
                    raise
        self.batch_data.update_epoch()
        self.key['LOSS'] = self.batch_data.epoch_loss.val
        self.key['EPOCH'] = self.epoch
        self.key['ITERATIONS'] = self.absolute_iter
        self.__update_db__()
        self.train_writer.add_scalar('loss_epoch',self.batch_data.epoch_loss.val,self.epoch)
        self.save_checkpoint()
        self.batch_data.batch_loss.reset()

    def validate_epoch(self):
        self.model.eval()
        with tqdm(self.val_loader, desc='Validation: [{0}/{1}]'.format(self.epoch, self.EPOCHS)) as pbar:
            for gt, inputs, visualization in pbar:

                inputs = [i.to(self.main_device) for i in inputs] if isinstance(inputs, list) else inputs.to(
                    self.main_device)
                gt = [i.to(self.main_device) for i in gt] if isinstance(gt, list) else gt.to(self.main_device)
                with torch.no_grad():
                    output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                    loss = self.criterion(output, gt)
                    self.tensorboard_writer(loss.data.item(),output,gt,self.absolute_iter,visualization)
        self.model.train()


    def _test(self):
        self.batch_data = ptutils.timers()
        ptutils.setup_logger('test_log', os.path.join(self.workdir, 'test_log.txt'), to_console=False)
        self.test_logger = logging.getLogger('test_log')
        #self.writer = SummaryWriter(log_dir=os.path.join(self.workdir, 'tensorboard'))

        batch_id = 0
        loss=0
        self.test_size = len(iter(self.test_loader))
        self.model.eval()
        with tqdm(self.test_loader) as pbar:
            for gt, inputs, visualization in pbar:

                inputs = [i.to(self.main_device) for i in inputs] if isinstance(inputs, list) else inputs.to(
                    self.main_device)
                gt = [i.to(self.main_device) for i in gt] if isinstance(gt, list) else gt.to(self.main_device)
                with torch.no_grad():
                    output = self.model(*inputs) if isinstance(inputs, list) else self.model(inputs)
                    loss += self.criterion(output, gt).item()
                #self.test_collector.collect(output,inputs,visualization)
                self.test_collector.collate(output, gt)
                batch_id += 1

                self.batch_data.test_logger(loss, self.test_logger)
    def __atribute_assertion__(self):
        assert hasattr(self, 'assertion_variables')
        try:
            for variable in self.assertion_variables:
                assert hasattr(self, variable) 
        except:
            raise Exception ('Variable assertion failed. Framework requires >{0}< to be defined'.format(variable))
    def __initilize_layers__(self):
        if self.inizilizable_layers is None:
            print('Network not automatically initilized')
        else:
            print('Network automatically initilized ')
            map(self.init_function,self.inizilizable_layers)

    def _train(self):
        self.training = True
        if self.resume:
            self._loadcheckpoint()
        else:
            if self.workname is None:
                self.__setup_experiment__()
                self.__initilize_layers__()

                self.batch_data = ptutils.timers()
            else:
                path_to_weights = self.workname
                self.__setup_experiment__()
                
                self._load_model(path_to_weights)
                self.batch_data = ptutils.timers()                 
        if self.dataparallel:
            self.model = torch.nn.DataParallel(self.model)      
        self.train_writer = SummaryWriter(log_dir=os.path.join(self.workdir,'tensorboard'))
        self.val_writer = SummaryWriter(log_dir=os.path.join(self.workdir,'tensorboard'))
    def save_gradients(self,absolute_iter):
        grads = self.tracker.grad(numpy=True)
        grad_path = os.path.join(self.workdir,'gradient_tracking')
        if not os.path.exists(grad_path):
            os.makedirs(grad_path)
        grad_path = os.path.join(grad_path,'grad_{0}_{1:06d}.npy'.format(self.workname,absolute_iter))
        np.save(grad_path,grads)
    @set_training
    def train(self,args):
        NotImplementedError
    def tensorboard_writer(self,loss,output,gt,absolute_iter,visualization):
        if self.model.training:
            self.train_writer.add_scalar('loss',loss,absolute_iter)
    def infer(self,*inputs):
        NotImplementedError
    def hyperparameters(self):
        NotImplementedError
    def gradients(self):
        pass
    

def test():
    import torch.utils.data
    class toy_example(torch.nn.Module):
        def __init__(self):
            super(toy_example,self).__init__()
            self.module1 = torch.nn.Conv2d(1,10,3)
            self.module2 = torch.nn.Conv2d(10,10,3)
        def forward(self,x):
            x = self.module1(x)
            return self.module2(x)
    class db(torch.utils.data.Dataset):
        def __len__(self):
            return 30
        def __getitem__(self,idx):
            return torch.rand(10,6,6),[torch.rand(1,10,10)],[]
    class toy_fw(pytorchfw):
        def hyperparameters(self):
            self.hihihi = 5
            self.initilizer = 'xavier'
            self.EPOCHS = 10
            self.batch_size = 2
            self.LR = 1
            #Def optimizer self.optimizer
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.LR)
            #Def criterion self.criterion
            self.criterion = torch.nn.L1Loss().to(self.main_device)
            self.dataparallel = False
        @set_training
        def train(self):
            datab = db() 
            self.train_loader= torch.utils.data.DataLoader(datab,batch_size=self.batch_size)
            for self.epoch in range(self.start_epoch,self.EPOCHS):
                self._train_epoch(self.train_iter_logger)
    fw = toy_fw(toy_example(),'./','/home/jfm/GitHub/flerken/flerken/framework/19cfddf/checkpoint.pth',0,False)
    return fw
        