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
    'vocab_size' : None


}

with open(f"{params['data_csv']}special_tokens", "rb") as fp:   #Pickling
    params['special_tokens'] = pickle.load(fp)