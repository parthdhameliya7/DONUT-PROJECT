import pickle

params = {

    #DATA PATH SETUP 
    'data_dir' : 'data/',

    'train_data_dir' : 'train_dataset/',
    'valid_data_dir' : 'valid_dataset/',
    'test_data_dir' : 'test_dataset/',
    'data_csv' : 'data_csv/',
    
    'train_csv' : 'train_dataset.csv',
    'valid_csv' : 'valid_dataset.csv',
    'test_csv' : 'test_dataset.csv',

    #IMAGE PARAMS 
    'image_size' : (615, 475),
    'aug_p' : 0.3,

    #TARGET PROCESSOR 
    'ignore_id' : -100,
    'max_length' : 410,
    'start_token' : '<s-donut>',
    'end_token' : '</s-donut>',
    'vocab_size' : None,

    #OPTIMIZER AND SCHEDULER
    'lr' : 1e-5,
    'eta_min' : 1e-6,
    'T_0' : 15,
    'step_scheduler_after' : 'epoch',
    #on_eval_loss

    #DATALOADER PARAMS
    'pin_memory' : True,
    'num_workers' : 12,

    #FIT PARAMS
    'epochs' : 15,
    'train_bs' : 64,
    'valid_bs' : 16,


}

with open(f"{params['data_csv']}special_tokens", "rb") as fp:   #Pickling
    params['special_tokens'] = pickle.load(fp)