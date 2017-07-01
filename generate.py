import torch
from model import *
import json
import pdb
import numpy as np
from build_dataset import *
import pdb


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"


with open('vocabulary.json') as json_data:
        word_to_int = json.load(json_data)
        #pdb.set_trace()
        int_to_word = dict(zip(list(word_to_int.values()), list(word_to_int.keys())))
X, _, _ = build_dataset()        
def generate(model_ft, outfile):

        with open(outfile, 'w') as in_file:
                start = np.random.randint(0, len(X)-1)
                pattern = X[start]
                pattern = pattern
                initial_pattern = pattern
                #in_file.write(' '.join(initial_pattern)+'\n')
                result = []
                print ("Seed:")
                #pdb.set_trace()
                print ("\"", ' '.join([int_to_word[value] for value in pattern]), "\"")
                query_text = str(' '.join([int_to_word[value] for value in pattern]))
                #print(query_text)
                in_file.write('Query:'+'\n'+query_text+'\n')
                model_ft.eval()
                hidden = model_ft.init_hidden()
                for i in range(200):
                        x = np.reshape(pattern, (1, len(pattern)))
                        
                        x = torch.from_numpy(x).long().cuda()
                        prediction = model_ft(Variable(x), hidden)
                        prediction = prediction.data.max(1)[1] #gives a tensor value
                        prediction = (prediction.cpu().numpy().item()) #converts that tensor into a numpy array 
                        result.append(int_to_word[prediction])
                        #pdb.set_trace()
                        pattern.append(prediction)
                        pattern = pattern[1:len(pattern)]
                print ("\nGenerated Sequence:")
                print (' '.join(result))
                generated_text = ' '.join(result)+'\n'
                in_file.write(generated_text)
                print ("\nDone.")   

if __name__ == '__main__':
    # Parse command line arguments
        import argparse
        argparser = argparse.ArgumentParser()
        argparser.add_argument('filename', type=str)
        argparser.add_argument('outfile', type=str)
        args = argparser.parse_args()
        model_ft = torch.load(args.filename)
        del args.filename
        (generate(model_ft, args.outfile))

