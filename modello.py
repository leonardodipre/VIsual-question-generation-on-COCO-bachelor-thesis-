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



    #curr_caption è un token iniziale, a cui poi ne deriva 5 e
    def beam( self, curr_captions ,token_current, prob_current, states , x, beam_width):
        
        
        next_captions = []
        
        hiddens, states = self.decoderRNN.lstm(curr_captions, states)
        hiddens = torch.cat((hiddens, x), dim=2)

        output = self.decoderRNN.linear(hiddens.squeeze(0))
        probs, indices = torch.topk(output, beam_width)
        

        for i in range(beam_width):
            #print("Indice", indices[0][i].item())
            
            
           

            start_tok = self.decoderRNN.embed(indices[0][i]).unsqueeze(0).unsqueeze(0)

            hiddens, states = self.decoderRNN.lstm(start_tok, states)       
            hiddens = torch.cat((hiddens, x),dim=2)


            output = self.decoderRNN.linear(hiddens.squeeze(0))
            probs_in, indices_in = torch.topk(output, beam_width)
           
            
            for j in range(beam_width):
                #predico lo statok per tutte le coppie nuove trovate
                start_tok = self.decoderRNN.embed(indices_in[0][j]).unsqueeze(0)
               
                indice = indices_in[0][j].item()
              

                prb = probs_in[0][j].item()
                #print("     Prob", prb)

                #print(start_tok.shape)
                #indice di partenza piu indice nuovo
                
                indice_next = token_current +[ indices[0][i].item(), indice ] 
                prb_next = probs[0][i].item() + prb + prob_current
                #print(indice_next)
                #print(prb_next)
                #print("##########")
                #curr_captions.append([[start_tok, indice_next, prb_next]]) 
                next_captions.append([start_tok,  indice_next , prb_next])

            #print("\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\")

      
        
        curr_captions = sorted(next_captions, key=lambda x: x[2], reverse=True)[:beam_width]

        return curr_captions, states


    def beam_loop(self, curr_captions , states , x, beam_width, vocabulary , loop):
        
        for _ in range(loop):
            print("Inizio loop ")
            tmp_caption = []

            stato = states

            for i in range(len(curr_captions)):
                    
                #print(curr_captions[i][1])
                #print(curr_captions[i][2])
                        
                tmp , states = self.beam( curr_captions[i][0].unsqueeze(0) ,curr_captions[i][1], curr_captions[i][2] , stato , x, beam_width)
                stato = states
                for i in range(len(tmp)):
                    tmp_caption.append([tmp[i][0],  tmp[i][1] , tmp[i][2]])
                
            for l in range(len(tmp_caption)):
                print(tmp_caption[l][1])
                print(tmp_caption[l][2])
                print( [vocabulary.itos[idx] for idx in tmp_caption[l][1]])
            print("-----------------------")

            tmp_caption = sorted(tmp_caption, key=lambda x: x[2], reverse=True)[:beam_width]



            curr_captions = tmp_caption
                
                
            print("-------------op 5-----------")
            for k in range(len(curr_captions)):
                print(curr_captions[k][1])
                print(curr_captions[k][2])
                print( [vocabulary.itos[idx] for idx in curr_captions[k][1]])



    def Beam_search(self, image, vocabulary, max_length=50, beam_width=15):
        
        result_captions = []
        
        self.eval()
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)

            states = x
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)

            curr_captions = [[start_tok, [1], 0.0]]


            curr_captions, states = self.beam( curr_captions[0][0] ,curr_captions[0][1], curr_captions[0][2] , states , x, beam_width)

            self.beam_loop( curr_captions , states , x, beam_width, vocabulary , loop=2)
         




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

            for _ in range(1):

                for _ , (token, sequence, score) in enumerate(result_caption):
                    
                    num_iter += 1
                    hiddens, states = self.decoderRNN.lstm(token, states)
                    hiddens = torch.cat((hiddens, x),dim=2)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))
                
                
               
                    probs_in, indices_in = torch.topk(output, beam_width)
                    
                    tmp_caption = []

                    for j in range(beam_width):
                        

                        indec = torch.tensor([indices_in[0][j].item()]).to(device)

                        start_tok = self.decoderRNN.embed(indec).unsqueeze(0)
                        
                        label = sequence + [indices_in[0][j].item()]
                        #print(label)
                        #print( [vocabulary.itos[idx] for idx in label])
                        sum_prob = probs_in[0][j].item() + score

                        if vocabulary.itos[indec.item()] == "<EOS>":
                            tmp_caption.append([start_tok , sequence , sum_prob])

                            result.append([vocabulary.itos[idx] for idx in sequence])
                            eos += 1
                            
                            if eos == beam_width:
                                return result
                            
                        else:
                            tmp_caption.append([start_tok ,label , sum_prob])
                   
                    



                    if num_iter == 1:
                        
                        print("result caption")
                        for i in range(len(result_caption)):
                            print(result_caption[i][1])
                            print(result_caption[i][2])


                        print("First iter")
                        for i in range(len(tmp_caption)):
                            print(tmp_caption[i][1])
                            print(tmp_caption[i][2])

                            result_caption.append([tmp_caption[i][0], tmp_caption[i][1] ,tmp_caption[i][2]])
                        print("\n\n")


                    else:
                        print("result caption")
                        for i in range(len(result_caption)):
                            print(result_caption[i][1])
                            print(result_caption[i][2])


                        print("N iter", num_iter)
                        for i in range(len(tmp_caption)):
                            print(tmp_caption[i][1])
                            print(tmp_caption[i][2])

                            result_caption.append([tmp_caption[i][0], tmp_caption[i][1] ,tmp_caption[i][2]])
                        print("\n\n")



                        if num_iter==3:
                            exit()
                            
                            
                
                





    def caption_image_beam_search(self, image, vocabulary, max_length=50, beam_width=5):
        result_captions = []
        
        self.eval()
        with torch.no_grad():
            x = self.encoderCNN(image).unsqueeze(0)

            states = x
            start_tok = self.decoderRNN.embed(torch.tensor([vocabulary.stoi["<SOS>"]]).cuda()).unsqueeze(0)

            curr_captions = [[start_tok, [], 0.0]]

            #caption, sequence, score = enumerate(curr_captions)
            """
            print("token ")
            print(curr_captions[0][0])
            print("sequence ")
            print(curr_captions[0][1])
            print("Score ")
            print(curr_captions[0][2])
            """
            
            #do il startoken in questo caso andra poi in un cilo
            #self.beam( curr_captions[0][0] , states , x, beam_width)
          
            
            """
            hiddens, states = self.decoderRNN.lstm(caption, states)
            hiddens = torch.cat((hiddens, x), dim=2)
            
            output = self.decoderRNN.linear(hiddens.squeeze(0))
                    
            probs, indices = torch.topk(output, beam_width)
            print(curr_captions)
            
            for step in range(50):
                next_captions = []
                
                
                for i, (caption, sequence, score) in enumerate(curr_captions):
                    hiddens, states = self.decoderRNN.lstm(caption, states)
                    hiddens = torch.cat((hiddens, x), dim=2)
                    output = self.decoderRNN.linear(hiddens.squeeze(0))
                    #predicted = output.argmax(1)
                    probs, indices = torch.topk(output, beam_width)

                    print(probs)
                    print(indices)
                    for i in range(beam_width):
                        
                        new_seq = torch.cat((caption, indices[0][i].unsqueeze(0)), dim=1)
                        new_score = score + probs[0][i].item()
                        new_tokens = score + [indices[0][i].item()]



                        if vocabulary.itos[probs.item()] == "<EOS>":
                            result_captions.append([sequence + [probs.item()], score + output[0][probs.item()]])
                        else:
                            next_captions.append([self.decoderRNN.embed(probs).unsqueeze(0), 
                                                    sequence + [probs.item()],
                                                    score + output[0][probs.item()]])
                            print("Next caption " ,next_captions)

                curr_captions = sorted(next_captions, key=lambda x: x[2], reverse=True)[:beam_width]
                
        result_captions = sorted(result_captions, key=lambda x: x[1], reverse=True)
        print( [ [vocabulary.itos[idx] for idx in cap[0]] for cap in result_captions ])
        """
