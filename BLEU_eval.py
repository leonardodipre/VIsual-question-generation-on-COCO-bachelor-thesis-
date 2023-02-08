import pandas as pd
import torch
from torch import nn, optim
from dataloader import get_loader
import torchvision.transforms as transforms
import json
from torchmetrics import BLEUScore
from eval import eval2
from eval import eval1
from  modello  import CNNtoRNN
from tqdm import tqdm
import torchvision.transforms as transforms
from save_model import save_checkpoint, load_checkpoint

from torch.nn.utils.rnn import pack_padded_sequence
import json


def blue_eval(preds_machine, target_human):
    
    weights_1 =(1.0/1.0, 0,0,0)
    weights_2 = (1.0/2.0, 1.0/2.0, 0,0)
    weights_3= (1.0/3.0, 1.0/3.0, 1.0/3.0,0 )
    weights_4 = (1.0/4.0, 1.0/4.0, 1.0/4.0 , 1.0/4.0)


    bleu_2 = BLEUScore(2, weights_2)
    bleu_3 = BLEUScore(3, weights_3)
    bleu_4 = BLEUScore(4, weights_4)

    return bleu_2(preds_machine, target_human), bleu_3(preds_machine, target_human), bleu_4(preds_machine, target_human)



def evaluation():
    preds_machine= ['No match in this phrase', 'the cat is on the tabel', ]
    target_human= [[ 'Complete correlation in this phrase ', 'the cat is on the tabel']]

    print(blue_eval(preds_machine, target_human))

    ###########################################################################################################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Declare transformations (later)
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ###########################################################################################################
                            #Dataset per Vacab
                                                        #DIRECTORY#
   
    csv = r'train_coco.csv'
    imm_dir =r'D:\Leonardo\Datasets\Coco\train2014\train2014'


    freq_threshold = 4 # 4019 vocab

    ############################################################################################################################################

    #Carico il dataset per i vocaboli presenti in train.csv

    _, dataset_vocab = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=2,
        )
    
    #########################################################################################
                        #parte per validation_coco.csv
       #Dir cartelle
    csv = "validation_coco.csv"              
    file_csv =pd.read_csv(csv)         # ID IMMAGINI
    dir_loc = imm_dir =r'D:\Leonardo\Datasets\Coco\val2014\val2014'
       
    #Carico la lista
    id_imm_list = file_csv["id"]
    questions_list = file_csv["Domande"]
    cocolist = file_csv["immagine_id"]
##################################################################################################




    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset_vocab.vocab)
    num_layers = 1
    learning_rate = 1e-4

    


    ###########################################################################

        # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)


    PATH = "save_Model"
    model.load_state_dict(torch.load(PATH))

    model.eval()


    for i in range(len(id_imm_list)):
        list_ = []
        for original_string in questions_list[i].split(','):
            
            characters_to_remove = "[]'"
            
            new_string = original_string
            
            for character in characters_to_remove:
                new_string = new_string.replace(character, "")
            
            #print("Strig " , new_string) 
            list_ +=[new_string]

        prediction, question = eval2(model,device, dataset_vocab, dir_loc, cocolist[i], list_ )

        
        question = [list_]
        pre_parantesist = [prediction]

        print("valuto immagine indice ", i)
        print(question)
        print(prediction)
        print(blue_eval(pre_parantesist, question))
        print("\n\n")
        
        

if __name__ == "__main__":
    evaluation()