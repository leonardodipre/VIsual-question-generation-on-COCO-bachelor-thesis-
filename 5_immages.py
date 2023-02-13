import pandas as pd
import torch
from torch import nn, optim
from dataloader import get_loader
import torchvision.transforms as transforms
import json
from torchmetrics import BLEUScore

from eval import eval1 , beam_search , eval2 
from  modello  import CNNtoRNN

import torchvision.transforms as transforms


from torch.nn.utils.rnn import pack_padded_sequence
import json


def blue_eval(preds_machine, target_human):
    
    weights_1 =(1.0/1.0, 0,0,0)
    weights_2 = (1.0/2.0, 1.0/2.0, 0,0)
    weights_3= (1.0/3.0, 1.0/3.0, 1.0/3.0,0 )
    weights_4 = (1.0/4.0, 1.0/4.0, 1.0/4.0 , 1.0/4.0)

    bleu_1 = BLEUScore(1, weights_1)
    bleu_2 = BLEUScore(2, weights_2)
    bleu_3 = BLEUScore(3, weights_3)
    bleu_4 = BLEUScore(4, weights_4)

    return bleu_1(preds_machine, target_human).item(), bleu_2(preds_machine, target_human).item(), bleu_3(preds_machine, target_human).item(), bleu_4(preds_machine, target_human).item()



def evaluation():
    

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
   
    csv = r'csv_file/train_coco.csv'
    imm_dir =r'D:\Leonardo\Datasets\Coco\train2014\train2014'


    freq_threshold = 4 # 4019 vocab

    ############################################################################################################################################

    #Carico il dataset per i vocaboli presenti in train.csv

    _, dataset_vocab = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=1,
        )
    
##################################################################################################




    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset_vocab.vocab)
    num_layers = 1
    learning_rate = 1e-4

    


    ###########################################################################

        # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)

    for j in range(5,6):
        
        PATH = r"D:\Leonardo\VQG_final\modelli\save_Model" + str(j)
        print("Read ", PATH)
        model.load_state_dict(torch.load(PATH))

        model.eval()
        
        print("IMM 1")
        a =[['is this in a museum?', 'how many animals are in the picture?', 'what kind of animal is shown?']]
        pred =beam_search(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000136.jpg")
        #pred= beam_search(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000192.jpg")
        print(pred)
        


        """
        print("IMM 1")
        a =[['is this in a museum?', 'how many animals are in the picture?', 'what kind of animal is shown?']]
        pred =eval1(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000136.jpg")
        print(a)
        print(pred)
        pred_ = [pred]

        print(blue_eval(pred_, a))
        print("\n")

        
        print("IMM 2")
        a = [['what sport is being played?', 'is the catcher wearing safety gear?', 'what is the name of the teams?']]
        pred= eval1(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000192.jpg")
        print(a)
        print(pred)
        pred_ = [pred]
        print(blue_eval(pred_, a))
        print("\n")

        print("IMM 3")
        a = [['what color is the car on the far right?', 'is there an ice cream truck?', 'is it daytime?', 'is the dog real?', 'what shoe company is advertised?', 'what color is the truck?']]
        pred = eval1(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000000257.jpg")
        print(a)
        print(pred)
        pred_ = [pred]
        print(blue_eval(pred_, a))
        print("\n")

        print("IMM 4")
        a = [['is the horse wearing something on its ankles?', 'how many horses are there?', 'is the woman riding english or western saddle?', 'is this in color or black and white?', 'what type of pants is the rider wearing?', 'is this a full grown horse?', 'is the woman wearing a long sleeve, short sleeve, or sleeveless blouse?', 'is one horse riderless?', 'is this horse wearing a saddle?']]
        pred = eval1(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000003209.jpg")
        print(a)
        print(pred)
        pred_ = [pred]
        print(blue_eval(pred_, a))
        print("\n")

        print("IMM 5")
        a = [['are there a lot of cars parked on the street?', 'what is cast?', 'what is parked beside the curb?']]
        pred =eval1(model, device, dataset_vocab, "D:\Leonardo\VQG_final\immagini5\COCO_val2014_000000285742.jpg")
        print(a)
        print(pred)
        pred_ = [pred]
        print(blue_eval(pred_, a))
        print("\n")
        """
if __name__ == "__main__":
    evaluation()