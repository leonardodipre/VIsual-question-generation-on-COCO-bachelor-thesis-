import pandas as pd
import torch
from torch import nn, optim
from  modello  import CNNtoRNN
from dataloader import get_loader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.nn.utils.rnn import pack_padded_sequence
from eval import eval1




def train():


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    writer = SummaryWriter("runs/coco_vqg")
    # Declare transformations (later)
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            # transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
        ]
    )

    ############################################################################################################################################
                                                        #DIRECTORY#
    
    
    csv = r'csv_file/train_coco.csv'
    imm_dir =r'D:\Leonardo\Datasets\Coco\train2014\train2014'


    freq_threshold = 4 # 4019 vocab

    ############################################################################################################################################


    loader, dataset = get_loader(
        csv, imm_dir, freq_threshold, transform=transform , num_workers=2,
        )
  

    # Hyperparameters
    embed_size = 224
    hidden_size = 224
    vocab_size = len(dataset.vocab)
    num_layers = 1
    learning_rate = 1e-4
    num_epochs = 20
    train_CNN = False
    load_model = False
    save_model = True
    ###########################################################################

    # Model declaration
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers).to(device)
    
    # Criterion declaration
    weight = torch.ones(vocab_size).to(device)
    weight[0] = 0 # Ignore the padding
    weight[3] = 0 # Ignore the unk token
    criterion = nn.CrossEntropyLoss(weight=weight)
    # Optimizer declaration
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for name, param in model.encoderCNN.resNet.named_parameters():
        # Only finetune the CNN
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

   

    j = 0
    
    for epoch in range(num_epochs):
        
        
        PATH = "save_Model" + str(j) 
        print(PATH)
        torch.save(model.state_dict(), PATH)
        j += 1
        epoch_loss = 0

        step = 0 
        step_loss = 0 
       
        for  i, (imgs, questions, lengths, index) in  enumerate(tqdm(loader)):

            model.train()

            imgs = imgs.to(device)
            questions = questions.to(device)
    
            outputs = model(imgs,questions,lengths)

            targets = pack_padded_sequence(questions[:, 1:], [l-1 for l in lengths], batch_first=True)[0]

            loss = criterion(outputs, targets)

            epoch_loss += loss.item()
            step_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            step+=1

            
            if(step%150 == 0 and step!=0):
                print(f"Step{step}/Epoch{epoch}], Loss: {step_loss/150:.4f}")
                writer.add_scalar("Looss" , step_loss/150 , step )

                with open("modelli\loss_valuesVQG_1_Final.txt", "a") as f:
                    f.write("Epoch: " + str(epoch) + ", Loss: " + str(step_loss/150) + "\n")
   
                step_loss = 0

            
            
                

        
       

    writer.close()

if __name__ == "__main__":
    train()



