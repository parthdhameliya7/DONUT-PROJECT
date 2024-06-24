import pickle

params = {

    #DATA PATH SETUP 
    'data_csv' : 'csvs/',

    'model_path' : 'models/donut_model_0_0_1.pt',
    'save_best_model' : 'on_eval_metric',
    'save_on_metric' : 'edit_distance',
    'save_model_at_every_epoch' : False,

    'train_csv' : 'train_dataset_2024-06-22-19:39:00.csv',
    'valid_csv' : 'valid_dataset_2024-06-22-19:39:00.csv',
    'test_csv' : 'test_dataset_2024-06-22-19:39:00.csv',

    #IMAGE PARAMS 
    'image_size' : (800, 550),
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
    'lr' : 1e-3,
    'eta_min' : 1e-6,
    'T_0' : 20,
    'step_scheduler_after' : 'epoch',
    #on_eval_loss

    #DATALOADER PARAMS
    'pin_memory' : True,
    'num_workers' : 12,

    #FIT PARAMS
    'epochs' : 20,
    'train_bs' : 12,
    'valid_bs' : 12,


}

with open(f"special_tokens", "rb") as fp:   #Pickling
    params['special_tokens'] = pickle.load(fp)