import random

def split_datasets(total_file_num,save_path,train_val_ratio=0.5):
    ###
    # Inputs
    # total_file_num : the numbers of your own 3DOD dataset files
    # train_val_ratio : The ratio to divide train dataset and valid dataset.
    # save_path : the path that save the result txt files.

    #Outputs:
    #train.txt, val.txt,trainval.txt

    train_split_num = int(total_file_num*train_val_ratio)
    num_samples = [i for i in range(total_file_num)]
    train_samples = random.sample(num_samples,train_split_num)
    val_samples = list(set(num_samples)-set(train_samples))

    train_txt = open(save_path+'train.txt','w')
    val_txt = open(save_path+'val.txt','w')
    trainval_txt = open(save_path+'trainval.txt','w')
    
    for i in train_samples:
        train_txt.write("%06d"%i +'\n')
    print("train file done!")

    for i in val_samples:
        val_txt.write("%06d"%i +'\n')
    print("val file done!")
        
    for i in num_samples:
        trainval_txt.write("%06d"%i +'\n')
    print("trainval file done!")
    return 0

path = '/home/cv1/hdd/poseRefine/'
split_datasets(22440,path)
    
