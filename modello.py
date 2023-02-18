import torch
import numpy as np
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F

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

    
    def caption_image_gredy(self, image, vocabulary, max_length=50):
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
                start_tok = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

                result_caption.append(predicted.item())

        return [vocabulary.itos[idx] for idx in result_caption]


    def caption_image_multinomial(self, image, vocabulary, max_length=50):
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
                
                
                output = F.softmax(output, dim = 1)
                predicted = torch.multinomial(output, num_samples=1)
                
               
            
        
                start_tok = self.decoderRNN.embed(predicted)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    
                    break



                result_caption.append(predicted.item())
                

        return [vocabulary.itos[idx] for idx in result_caption]



    def top_5_multinomial(self, resul_captio_25, beam_width , x, vocabulary, resul):
        

        tmp = []
        for _ , (token, seq , score, states) in enumerate(resul_captio_25):
                

                #print(seq)
                

                token = token.unsqueeze(0).unsqueeze(0)
                
                hiddens, states = self.decoderRNN.lstm(token, states)
                
                hiddens = torch.cat((hiddens, x),dim=2)
               
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                
                
                output = F.softmax(output, dim = 1)
                predicted = torch.multinomial(output, num_samples=beam_width, replacement=False)
               
                
                
                for i in range(beam_width):
                    
                    token = self.decoderRNN.embed(predicted[0][i])

                    indice = predicted[0][i].item()
                    probabilita = output[0][indice].item()

                    #print(indice)
                    #print(probabilita)

                    sequeza = seq + [indice]
                    prob = score + probabilita
                    
                    if vocabulary.itos[indice]   == "<EOS>" or vocabulary.itos[indice] == "?" :
                        resul.append([token, sequeza, prob, states])
                        
                    else:
                        tmp.append([token, sequeza, prob, states])

                    #print(token.shape)
                    #print(sequeza)
                    #print(prob)
                    #print("")


        tmp = sorted(tmp, key=lambda x: x[2], reverse=True)[:beam_width]
        return tmp
               

    def beam_search_multinomial(self, image, vocabulary, beam_width ,max_length=50)  :
        result_caption = []
        result = []
        self.eval()
        with torch.no_grad():

            x = self.encoderCNN(image).unsqueeze(0)

            states = x
            
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)
            
            hiddens, states = self.decoderRNN.lstm(start_tok, states)
                
            hiddens = torch.cat((hiddens, x),dim=2)
               
            output = self.decoderRNN.linear(hiddens.squeeze(0))
                
                
            output = F.softmax(output, dim = 1)
            predicted = torch.multinomial(output, num_samples=beam_width, replacement=False)
            result_caption = []

          

            for i in range(beam_width):
                
                token = self.decoderRNN.embed(predicted[0][i])

                indice = [predicted[0][i].item()] 
                probabilita = output[0][indice].item()
                result_caption.append([token, indice, probabilita, states ])        

            #print("primi 5 initi token")   
            #for i in range(len(result_caption)):
                #print(result_caption[i][1])
                #print(result_caption[i][2])

            for i in range(10):
                
                tmp  = self.top_5_multinomial(result_caption, beam_width, x , vocabulary, result)

                result_caption = tmp
                
                #for i in range(len(result_caption)):
                    #print(result_caption[i][1])
                    #print(result_caption[i][2])
                    #print( [vocabulary.itos[idx] for idx in result_caption[i][1]])
                if (result ==  beam_width):
                    break 
        
        return result