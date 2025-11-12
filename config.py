from pprint import pprint
import os
import setproctitle

class Config:
    name = 'LiteCrackSeg_TUT'
    
    gpu_id = '0'
    
    setproctitle.setproctitle("%s" % name)

    # path
    train_data_path = 'datasets/TUT/train/train.txt' #path to dataset train folder
    val_data_path = 'datasets/TUT/valid/valid.txt' #path to dataset valid folder


    checkpoint_path = 'checkpoints'
    log_path = 'log'
    saver_path = os.path.join(checkpoint_path, name)
    max_save = 20

    # visdom
    vis_env = 'LiteCrackSeg_TUT'
    port = 8097
    vis_train_loss_every = 40
    vis_train_acc_every = 40
    vis_train_img_every = 120
    val_every = 200


    epoch = 100
    pretrained_model = ''
    weight_decay = 0.01 


    lr = 1e-4  
    momentum = 0.9
    use_adam = True
    train_batch_size = 4
    val_batch_size = 4
    test_batch_size = 4

    acc_sigmoid_th = 0.5 
    pos_pixel_weight = 0


    # checkpointer
    save_format = ''
    save_acc = -1
    save_pos_acc = -1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

    def show(self):
        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')
