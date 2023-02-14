import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class CNNencoder(nn.Module):
    def __init__(self, embed_size, train_CNN=False):
        super(CNNencoder,self).__init__()
        self.train_CNN = train_CNN
        self.resNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.preprocess = ResNet18_Weights.DEFAULT.transforms()
        self.resNet.fc = nn.Linear(self.resNet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resNet(self.preprocess(images))
        return features

class RNNdecoder(nn.Module):
    def __init__(self,embed_size, hidde_size, vocab_size, num_layers):
        super(RNNdecoder, self).__init__()

        # look up matrik, that map the indix for output words
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=0)

        self.lstm = nn.GRU(embed_size, hidde_size, num_layers, batch_first=True)
        self.linear= nn.Linear(hidde_size*2, vocab_size) # Append image features to all hidden states

        self.dropout = nn.Dropout(0)

    def forward(self, features, questions, lengths):
        embeddings = self.dropout(self.embed(questions))
        packed = pack_padded_sequence(embeddings, [l-1 for l in lengths], batch_first=True)
        hiddens, _ = self.lstm(packed,features.squeeze().unsqueeze(0))
        hiddens = pad_packed_sequence(hiddens, batch_first=True)
        new_hiddens = torch.cat((hiddens[0], features.unsqueeze(1).expand(-1,hiddens[0].shape[1],-1)), dim=2)
        packed = pack_padded_sequence(new_hiddens, [l-1 for l in lengths], batch_first=True)
        outputs = self.linear(packed[0])
        return outputs

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = CNNencoder(embed_size)
        self.decoderRNN = RNNdecoder(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions, lengths):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions, lengths)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []
        
        self.eval()
        with torch.no_grad():

            x = self.encoderCNN(image).unsqueeze(0)

            

            states = x
            
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)
            
            for _ in range(max_length):

                hiddens, states = self.decoderRNN.lstm(start_tok, states)
                
                hiddens = torch.cat((hiddens, x),dim=2)
               
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                
                
               
                predicted = output.argmax(1)
                
                print(predicted.shape)
                start_tok = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    
                    break



                result_caption.append(predicted.item())
                

        return [vocabulary.itos[idx] for idx in result_caption]


    def top_5_beam(self ,x , states , probs_in, indices_in, beam_width , sequence, score, device , vocabulary,  result):

        #input 1 probs e un indice
        tmp_caption = []
       
        indec = torch.tensor([indices_in.item()]).to(device)

        start_tok = self.decoderRNN.embed(indec).unsqueeze(0)

        previus_label = sequence + [indices_in.item()]
        previus_score = score + probs_in.item()

       
        hiddens, states = self.decoderRNN.lstm(start_tok, states)
        hiddens = torch.cat((hiddens, x),dim=2)
        output = self.decoderRNN.linear(hiddens.squeeze(0))
                    
                    
        #prendo indice 1 e collego alle sue top 5 prbs
        probs_2, indices_2 = torch.topk(output, beam_width)

        #print("Indici nuove ", indices_2)


        for j in range(beam_width):
                            
           

            indec = torch.tensor([indices_2[0][j].item()]).to(device)
            start_tok = self.decoderRNN.embed(indec).unsqueeze(0)

            label = previus_label + [indices_2[0][j].item()]
            
           
            #print( [vocabulary.itos[idx] for idx in label])
            sum_prob = probs_2[0][j].item() + previus_score

            #print("New label", label)
            #print( [vocabulary.itos[idx] for idx in label])
            #print("New score", sum_prob)
           
            

            if vocabulary.itos[indec.item()] == "<EOS>":

                tmp_caption.append([start_tok , sequence , sum_prob])

                result.append([vocabulary.itos[idx] for idx in sequence])
                
                            
            else:

                tmp_caption.append([start_tok , label , sum_prob])


            
        return tmp_caption 


    def caption_image_Bean(self, image, vocabulary, beam_width, max_length=50):
        
        num_iter = 0
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #inserisco i risultati qunado trovo un <eos>
        result = []
        eos = 0

        self.eval()
        with torch.no_grad():

            x = self.encoderCNN(image).unsqueeze(0)
            states = x
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)
            
            #inserisco statok, no vocabili e punteggio 0.0
            result_caption = [[start_tok, [] , 0.0]]

            for _ in range(50):

                for k , (token, sequence, score) in enumerate(result_caption):

                    """ 
                    print(len(result_caption))
                    
                    print("Risultati")
                    for u in range(len(result)):
                        print(result[u])
                    """
        

                    hiddens, states = self.decoderRNN.lstm(token, states)
                    hiddens = torch.cat((hiddens, x),dim=2)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))
                    
                    
                
                    probs_in, indices_in = torch.topk(output, beam_width)
                    
                    tmp= []

                    for j in range(beam_width):
                        
                        tmp_1 = self.top_5_beam(x , states,  probs_in[0][j], indices_in[0][j], beam_width , sequence, score, device, vocabulary , result)
                       
                        if len(result)== beam_width:
                            exit()

                        for i in range(len(tmp_1)):
                            tmp.append([tmp_1[i][0], tmp_1[i][1], tmp_1[i][2]])

                    print("Iter k j", k , j)
                    for y in range(len(tmp)):
                        print(tmp[y][1])
                        print(tmp[y][2])
                        print( [vocabulary.itos[idx] for idx in tmp[y][1]])

                    print("\n\n")

                    tmp = sorted(tmp, key=lambda x: x[2], reverse=True)[:beam_width]
                    
                    print("TOP ")
                    for y in range(len(tmp)):
                        print(tmp[y][1])
                        print(tmp[y][2])
                        print( [vocabulary.itos[idx] for idx in tmp[y][1]])

                    print()


                    result_caption = tmp
                    for h in range(len(result_caption)):
                        a=1
                        #print(result_caption[h][1])
                        #print(result_caption[h][2])
                        #print( [vocabulary.itos[idx] for idx in result_caption[h][1]])

                       
                                
                            
                
                





 