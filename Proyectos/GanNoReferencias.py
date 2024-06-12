#!/usr/bin/env python

##Create conda environment
## check compatibilities: https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html

##    #https://pytorch.org/get-started/locally/
##    https://pytorch.org/get-started/previous-versions/

#    conda config --add channels conda-forge
#    conda config --set channel_priority strict

#    conda create -n tfmgan python=3.8 pip 
#    conda activate tfmgan
#    
#    pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118

#    pip install jupyter notebook
#    pip install future tensorflow
#    pip install future tensorboard
#    
#    pip install matplotlib
#    pip install scikit-learn
#    pip install pandas
#    pip install tqdm

#   conda activate tfmgan



#Example of use:
# ./test_traingan.py ./ImgFolder/ 5 10 ./OutTrain/

import sys

if __name__ == "__main__":
    argc = len(sys.argv)
    #print(argc)
    if argc <= 1:
        print("Usage: ./test_traingan.py inputfolder epochs batchsize outputfolder")
        print("Example: ./test_traingan.py ./ImgFolder/ 5 10 ./OutTrain/")
        sys.exit(0)

inputfolder = sys.argv[1]
nepochs = int(sys.argv[2])
batchsize = int(sys.argv[3])
outputfolder = sys.argv[4]

#do not display warnings
import warnings
warnings.filterwarnings('ignore')


import os
import random
from PIL import Image,  ImageOps
import numpy as np
#import tensorflow as tf
#from tensorflow.python.keras.layers import Input, Dense
from torchvision import models
import time
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as datasets
from torch.utils.data import Subset
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import copy
from sklearn.metrics import confusion_matrix
from torch.utils.data import RandomSampler, ConcatDataset
import pandas as pd
import shutil
from PIL import ImageFile
from tqdm import tqdm
from PIL import ImageEnhance
import torch.nn.init as init

 
if torch.cuda.is_available():
    device = torch.device("cuda")
    #print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("GPU no disponible, usando CPU")
    

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            
def initialize_model2(num_classes):
    #Los pesos si se entrenan
    model_ft = models.inception_v3(weights=False) #Se carga el modelo inception_v3 sin sus pesos preentrenados.
    # False en las redes convencionales, se entrenaran
    set_parameter_requires_grad(model_ft, False)
    # Se modifica la red auxiliar
    num_ftrs = model_ft.AuxLogits.fc.in_features
    model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)

    # Se modifica la red principal para que sea de dos clases con una salida softmax y coger la con mas probabilidad.
    model_ft.dropout= nn.Dropout(0.5)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs,num_classes),
        nn.Dropout(0.3),
        nn.Softmax(dim=1)
    )

    input_size=299 #Tamaño que usa la red
    return model_ft,input_size
    
    
class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()
        # Capas convolucionales
        self.conv1 = self.conv_block(in_channels, 32, kernel_size=4, stride=2)  # k4n32s2
        self.conv2 = self.conv_block(32, 64, kernel_size=4, stride=2)  # k4n64s2
        self.conv3 = self.conv_block(64, 128, kernel_size=4, stride=2)  # k4n128s2
        self.conv4 = self.conv_block(128, 256, kernel_size=4, stride=2)  # k4n256s2
        self.conv5 = self.conv_block(256, 256, kernel_size=4, stride=2)  # k4n256s2

        # Capas deconvolucionales con skip connections
        self.deconv6 = self.deconv_block(256, 256, kernel_size=4, stride=2)  # k4n256s2k3n256s1
        self.deconv7 = self.deconv_block(256*2, 128, kernel_size=4, stride=2)  # k4n128s2k3n128s1
        self.deconv8 = self.deconv_block(128*2, 64, kernel_size=4, stride=2)  # k4n64s2k3n64s1
        self.deconv9 = self.deconv_block(64*2, 32, kernel_size=4, stride=2)  # k4n32s2k3n32s1
        self.deconv10 = self.deconv_block(32*2, 32, kernel_size=4, stride=2)  # k4n32s2k3n32s1k3n3s1
        self.conv11 = nn.Conv2d(32, 3, 3, stride=1, padding=1, bias=False)
        self.initialize_weights()
    def conv_block(self, in_channels, out_channels, kernel_size, stride):
        
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def initialize_weights(self): # Se inician los pesos
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight, mean=0.0, std=0.02)  # Inicialización normal con media 0 y desviación estándar 0.02
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # Inicialización de sesgo a 0
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight, mean=1.0, std=0.02)  # Inicialización normal con media 1 y desviación estándar 0.02
                init.constant_(m.bias, 0)  # Inicialización de sesgo a 0

    def deconv_block(self, in_channels, out_channels, kernel_size, stride):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)
        conv4_out = self.conv4(conv3_out)
        conv5_out = self.conv5(conv4_out)

        deconv6_out = self.deconv6(conv5_out)
        deconv7_out = self.deconv7(torch.cat((conv4_out, deconv6_out), 1))  # Skip connection
        deconv8_out = self.deconv8(torch.cat((conv3_out, deconv7_out), 1))  # Skip connection
        deconv9_out = self.deconv9(torch.cat((conv2_out, deconv8_out), 1))  # Skip connection
        deconv10_out = self.deconv10(torch.cat((conv1_out, deconv9_out), 1))  # Skip connection
        conv11_out = self.conv11(deconv10_out)
        return torch.tanh(conv11_out)
        
class CustomDataset(Dataset):
    def __init__(self, ruta_mala, transform):
        
        self.ruta_mala = ruta_mala
        
        
        self.transform = transform

        #lista de nombres de archivos malas y buenas
        self.nombres_malas = os.listdir(ruta_mala)
        
        
    def __len__(self): # Devuelve la longitud
        
        return len(self.nombres_malas)

    def __getitem__(self, idx):
        nombre_mala = os.path.join(self.ruta_mala, self.nombres_malas[idx])
        
       
        imagen_mala = Image.open(nombre_mala)
        
        
        if self.transform:
            imagen_mala = self.transform(imagen_mala)
            
        
        return imagen_mala
        
           
################# MAIN ##########################

###### DISCRIMINADOR
Modeloft, _ = initialize_model2(2)
Discriminador = Modeloft.to(device)
Modeloft.aux_logits = False

# Construir la ruta completa al archivo
PATH = f'./Pesos/Discriminador.pt'

# Cargar los pesos del modelo desde el archivo
Discriminador.load_state_dict(torch.load(PATH))

#Necesario en inferencia para que siempre salga el mismo resultado (evita dropout, batchnorm, etc)
#Se utiliza en combinación con torch.no_grad()
Discriminador.eval()

nuevas_dimensiones = (299, 299) #input size for discriminator


###### GENERADOR
generator = Generator(3,3).to(device)

transform = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.ToTensor() #Transforma las imagenes a tensor
])

transform299 = transforms.Compose([
    transforms.Resize((299,299)),   
    transforms.ToTensor() #Transforma las imagenes a tensor
])



transform2 = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.RandomHorizontalFlip(p=1), #Realiza un flip horizontal a todas las imagenes
    transforms.ToTensor()
])


transform3 = transforms.Compose([
    transforms.Resize((256,256)),   
    transforms.RandomVerticalFlip(p=1), #Realiza un flip vertical a todas las imagenes
    transforms.ToTensor()
])

###### LOSSES
L2Loss = nn.MSELoss().to(device)
L1Loss = nn.L1Loss().to(device)
AdvLoss = nn.BCELoss().to(device)
#AdvLoss = nn.BCELoss(reduction='sum').to(device) #Why 'sum'? Not used in  https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html and https://github.com/nbertagnolli/pytorch-simple-gan/blob/master/train.py

Path_realesMalas = inputfolder

##Compute discriminator value for input image resized to (299, 299)
#conjunto_datos_real = CustomDataset(Path_realesMalas,transform=transform299)
#tamany_real = batchsize  # Un batch de 36 y se genera el dataLoader con un total de 26577 imagenes emparejadas
#loader_real = DataLoader(conjunto_datos_real, batch_size=tamany_real, shuffle=True) 
#dataloader_iterator = iter(loader_real)

#print("Original discriminator (input 299 x 299):")
#imagenesMalas = next(dataloader_iterator)
#imagenesMalas = imagenesMalas.to(device)
#with torch.no_grad():
#    discriminador_input = Discriminador(imagenesMalas)[:, 1]
#print(discriminador_input)


# Crear instancias de DatasetPix2Pix para cada conjunto de transformaciones: resized to (256, 256) 
conjunto_datos_real = CustomDataset(Path_realesMalas,transform=transform)

#conjunto_datos_real = CustomDataset(Path_realesMalas,transform=transform) + CustomDataset(Path_realesMalas, transform=transform2)  + CustomDataset(Path_realesMalas, transform=transform3) 

tamany_real = batchsize  # Un batch de 36 y se genera el dataLoader con un total de 26577 imagenes emparejadas
loader_real = DataLoader(conjunto_datos_real, batch_size=tamany_real, shuffle=True) 
print("Images in training dataset: %i"%(len(loader_real.dataset)))


## El mejor lr por ahora ha sido 0.001 no tocar !!!!!!!!!!!!!!
optimizador_generador = torch.optim.Adam(generator.parameters(), lr=0.001,betas=(0.5, 0.999)) 

eps = torch.finfo().eps

# Entrenar el generador
generator.train()

dataloader_iterator = iter(loader_real)

avgclassepoch= []
lossepoch = []

meanclassInput = 0
for epoch in range(nepochs):
    #REPEAT THIS PART FOR EACH EPOCH
    

    print("epoch %i"%(epoch))
    lossacum = 0
    meanclass = 0
    nclass = 0
    #REPEAT THIS PART FOR EACH BATCH

    ibatch=0
    for imagenesMalas in loader_real:
    
        optimizador_generador.zero_grad() #INSIDE THE ITERATOR LOOP! https://stackoverflow.com/questions/65570250/pytorch-mini-batch-when-to-call-optimizer-zero-grad
    
        #imagenesMalas = next(dataloader_iterator) #Use if you need to access batches one by one
        imagenesMalas = imagenesMalas.to(device)

        #compute discriminator output with (256, 256) input
        #print("input 256 x 256")
        if epoch == 0:
            with torch.no_grad():
                discriminador_input = Discriminador(imagenesMalas)[:, 1]
            #print(discriminador_input)

        #compute discriminator output with (256, 256) input resized to (299, 299) (bilinear interpolation)
#        print("input 299 x 299 from 256 x 256 (bilinear)")
#        inputimages = F.interpolate(imagenesMalas, size=nuevas_dimensiones, mode='bilinear', align_corners=False)
#        with torch.no_grad():
#            discriminador_input = Discriminador(inputimages)[:, 1]
#        print(discriminador_input)

        ##compute discriminator output with (256, 256) input resized to (299, 299) (bicubic interpolation)
        #print("input 299 x 299 from 256 x 256 (bicubic)")
        #inputimages = F.interpolate(imagenesMalas, size=nuevas_dimensiones, mode='bicubic', align_corners=False)
        #with torch.no_grad():
        #    discriminador_input = Discriminador(inputimages)[:, 1]
        #print(discriminador_input)

        ##compute discriminator output with (256, 256) input resized to (299, 299) (nearest-neighbor interpolation)
        #print("input 299 x 299 from 256 x 256 (nearest neighbor)")
        #inputimages = F.interpolate(imagenesMalas, size=nuevas_dimensiones, mode='nearest')
        #with torch.no_grad():
        #    discriminador_input = Discriminador(inputimages)[:, 1]
        #print(discriminador_input)


        #print("batch %i"%(ibatch))

        with torch.set_grad_enabled(True):  # Asegura que se registren las operaciones para gradientes
            generadas_malas = generator(imagenesMalas)

        generadas = F.interpolate(generadas_malas, size=nuevas_dimensiones, mode='bilinear', align_corners=False)
        #discriminador_salida = Discriminador(generadas.detach())[:, 1] #NEED DETACH? NO, since we want to compute generator loss, not discriminator loss. See: https://discuss.pytorch.org/t/couldnt-understand-how-detach-is-changing-the-generator-working/84637 and https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html and https://github.com/nbertagnolli/pytorch-simple-gan/blob/master/train.py

        discriminador_salida = Discriminador(generadas)[:, 1]
        #print(discriminador_salida)
        #print(torch.log(discriminador_salida))

        etiquetas_target = torch.zeros(imagenesMalas.size(0), dtype=torch.float, device=device) # Son 0 porque es lo que se quiere que el generador consiga
        adversarialLoss = AdvLoss(discriminador_salida, etiquetas_target)

        similarityLoss = L1Loss(imagenesMalas, generadas_malas)
        
#        if epoch < int(nepochs/2):
#            #similarityLoss = L2Loss(imagenesMalas, generadas_malas)
#            similarityLoss = L1Loss(imagenesMalas, generadas_malas)
#            loss = similarityLoss
#        else:
#            #etiquetas_target = torch.zeros(imagenesMalas.size(0), dtype=torch.float, device=device) # Son 0 por que es lo que se quiere que el generador consiga
#            #adversarialLoss = L2Loss(discriminador_salida,etiquetas_target)
#            #adversarialLoss = L1Loss(discriminador_salida,etiquetas_target)
#            #adversarialLoss = L1Loss(torch.log(discriminador_salida+eps), torch.log(etiquetas_target+eps))
#            adversarialLoss = torch.sum(discriminador_salida) 
#            loss = adversarialLoss           

##        if epoch == int(nepochs/2):
##            #increase learning rate
##            optimizador_generador = torch.optim.Adam(generator.parameters(), lr=0.001,betas=(0.5, 0.999)) 

#        #print("Loss=%2.4f"%(loss))

#        with torch.no_grad():
#            lossacum+=loss
        
        loss = adversarialLoss + similarityLoss
        #loss = similarityLoss
        

        if len(imagenesMalas) == tamany_real:
            print('Train Epoch: {} [{}/{}], advloss:{:.5f}, simloss:{:.5f}, loss:{:.5f}'
                  .format(epoch , ibatch * len(imagenesMalas), len(loader_real.dataset), adversarialLoss, similarityLoss, loss))
        else:
            print('Train Epoch: {} [{}/{}], advloss:{:.5f}, simloss:{:.5f}, loss:{:.5f}'
                  .format(epoch,  len(loader_real.dataset) - len(imagenesMalas) , len(loader_real.dataset), adversarialLoss, similarityLoss, loss))
        

        with torch.no_grad():
            if epoch == 0:
                meanclassInput += torch.sum(discriminador_input) 
            meanclass += torch.sum(discriminador_salida) 
            nclass += len(discriminador_salida)
            lossacum += loss

        
        
        loss.backward()
        optimizador_generador.step()


#        imout=generadas.detach()
#        for i in range(len(imout)):
#            nameout=outputfolder+'out_e'+str(epoch)+'_batch'+str(ibatch)+'_'+str(i)+'.png'
#            torchvision.utils.save_image(imout[i], nameout)

        ibatch+=1
    
    print("Loss acum epoch=%2.4f"%(lossacum))
    print("Avg. classification epoch=%2.4f"%(meanclass/nclass))
    
    avgclassepoch.append(meanclass/nclass)
    lossepoch.append(lossacum)
    
    
    with open(outputfolder+"/traininginfo.txt", "a+") as f:
        if epoch == 0:
            f.write("   Avg. classification Input=%2.4f\n"%(meanclassInput/nclass))        
        f.write("epoch %i:\n"%(epoch))
        f.write("   Loss acum epoch=%2.4f\n"%(lossacum))
        f.write("   Avg. classification epoch=%2.4f\n"%(meanclass/nclass))

    # Mostrar una imagen generada 
    # Ruta de la imagen de prueba
    #ruta_imagen_prueba = './PruebaImg/nm_1up.jpg'  
    ruta_imagen_prueba = './PruebaImg/n01496331_18538.jpg'  

    # Cargar la imagen de prueba y aplicar transformaciones para pasar a tensor
    imagen_prueba = Image.open(ruta_imagen_prueba)
    imagen_prueba = transform(imagen_prueba).unsqueeze(0) 
    imagen_prueba = imagen_prueba.to(device)
    with torch.no_grad():
        generator.eval()  # Poner el generador en modo de evaluación
        imagen_generada = generator(imagen_prueba)
    nameout=outputfolder+'out_e'+str(epoch)+'.png'
    torchvision.utils.save_image(imagen_generada, nameout)

    #store weights
    torch.save(generator.state_dict(), f'./Pesos/CheckPoint/generator{epoch}.pth')
    
#plot average classification per epoch
selected_ticks = [i for i in range(5,nepochs + 1,5)]
avgclassepoch_np = [tensor.detach().cpu().numpy()  for tensor in avgclassepoch]
plt.title("Puntuacion Discriminador en imagenes generadas")
plt.xlabel("Epocas")
plt.ylabel("Perdida")
plt.plot(range(1,nepochs+1),avgclassepoch_np[0:nepochs],label="Train")
plt.ylim(-0.5,1.5)

plt.xticks(selected_ticks)
plt.legend()
plt.savefig(outputfolder+"/avgclass.pdf", format="pdf", bbox_inches="tight")
plt.show()  
plt.close() 

np.savetxt(outputfolder+"/avgclass.txt", avgclassepoch_np, fmt='%2.4f') 
 

#plot loss per epoch
selected_ticks = [i for i in range(5,nepochs + 1,5)]
lossepoch_np = [tensor.detach().cpu().numpy()  for tensor in lossepoch]
plt.title("Perdida total en imagenes generadas")
plt.xlabel("Epocas")
plt.ylabel("Perdida")
plt.plot(range(1,nepochs+1),lossepoch_np[0:nepochs],label="Train")
plt.ylim(0,max(lossepoch_np).item()*1.1)

plt.xticks(selected_ticks)
plt.legend()
plt.savefig(outputfolder+"/loss.pdf", format="pdf", bbox_inches="tight")
plt.show()  
plt.close()  
    
np.savetxt(outputfolder+"/loss.txt", lossepoch_np, fmt='%2.4f') 
    
    
    
    
    

