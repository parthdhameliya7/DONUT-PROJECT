import pickle

params = {

    #DATA PATH SETUP 
    'data_dir' : 'data/',

    'train_data_dir' : 'train_dataset/',
    'valid_data_dir' : 'valid_dataset/',
    'test_data_dir' : 'test_dataset/',
    'data_csv' : 'data_csv/',

    'model_path' : 'models/donut_model_0.0.1.pt',
    'save_best_model' : 'on_eval_metric',
    'save_on_metric' : 'edit_distance',
    'save_model_at_every_epoch' : False,

    'train_csv' : 'train_dataset.csv',
    'valid_csv' : 'valid_dataset.csv',
    'test_csv' : 'test_dataset.csv',

    #IMAGE PARAMS 
    'image_size' : (900, 650),
    'mean' : [0.485, 0.456, 0.406],
    'std' : [0.229, 0.224, 0.225],
    'aug_p' : 0.3,

    #TARGET PROCESSOR 
    'ignore_id' : -100,
    'max_length' : 410,
    'start_token' : '<s-donut>',
    'end_token' : '</s-donut>',
    'vocab_size' : None,

    #OPTIMIZER AND SCHEDULER
    'lr' : 1e-4,
    'eta_min' : 1e-6,
    'T_0' : 30,
    'step_scheduler_after' : 'epoch',
    #on_eval_loss

    #DATALOADER PARAMS
    'pin_memory' : True,
    'num_workers' : 12,

    #FIT PARAMS
    'epochs' : 30,
    'train_bs' : 8,
    'valid_bs' : 8,


}

with open(f"{params['data_csv']}special_tokens", "rb") as fp:   #Pickling
    params['special_tokens'] = pickle.load(fp)