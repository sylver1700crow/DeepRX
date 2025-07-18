import torch
import numpy as np
from config import get_config
from trainer.load_data import load_data
from model.mmsenet import mmsenet

# 1. Charger la config
args = get_config()
args.phase = 'test'
args.test_epoch = 'epoch4'   # Mets le nom de ton fichier de checkpoint ici
args.data_mode = 'CDL-A'     # Adapter selon ton dataset
args.doppler = 75            # Adapter si besoin
args.ts_snr = 0  


# 2. Construire le modèle comme à l'entraînement
model = mmsenet(args)

# Charger le bon checkpoint
ckp_path = 'C:/Users/nzide/Desktop/DeepRX/result/MIMO/ckp/epoch4'
checkpoint = torch.load(ckp_path, map_location='cpu')
model.net.load_state_dict(checkpoint['model'])
model.net.eval()



# 3. Charger les données de test (ici un batch, tu adaptes si tu veux un seul exemple ou tout le testset)
input_x, sig_list, dop_list, Ideal_X, Ideal_H = load_data(
    dop=args.doppler,
    snr=[args.ts_snr],
    data_per=1.0,           # 1.0 pour tout le dataset, ou <1.0 pour un sous-échantillon
    phase='test',
    dataset_name=[args.data_mode],
    RX_Ants_list=[args.rx_ants]
)
# input_x shape typique: (nb, channel, 12, 24)
# sig_list shape: (nb, 1, 1)
# dop_list shape: (nb, 1, 1)


# Si tu veux juste un batch particulier :
input_x = input_x[:1]
sig_list = sig_list[:1]
dop_list = dop_list[:1]


# 4. Faire la prédiction
with torch.no_grad():
    # Si le modèle attend des dimensions supplémentaires (ex : .to(device)), adapte ici
    pred_llr = model.net(input_x, sig_list, dop_list)
    # Le format de retour dépend de ta définition du forward, vérifie la doc/model.py si doute
    # Souvent ça donne un tuple (H1, LLR), donc :
    # H1, LLR = model.net(input_x, sig_list, dop_list)
    # print(LLR.shape)
print(pred_llr)  # Affiche tes prédictions (LLR de bits)




'''

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import matplotlib.pyplot as plt

# Charger modèle
model = ...  # ton modèle MMSENet déjà instancié et chargé
model.eval()

# Sélectionne la couche cible
target_layers = [model.net.rs7]  # Adapter selon ton MMSENet

# Crée Grad-CAM
cam = GradCAM(model=model.net, target_layers=target_layers, use_cuda=True)

# Prépare input
input_tensor = ...  # format (1, C, H, W) par exemple (1, 2, 72, 800)

# Applique Grad-CAM
grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(0)])
grayscale_cam = grayscale_cam[0, :]

# Visualisation brute
plt.imshow(grayscale_cam)
plt.colorbar()
plt.show()

# je dois interpréter la heatmap comme un poids spatial sur les features qui ont mené aux LLR :
# Cela t'indique quelles zones de l’entrée (en temps-fréquence par ex.) ont contribué à la décision de bit."

# Grad-CAM sur rs7	Feature riche, avant réduction finale	Balance entre contexte et détail
# Grad-CAM sur rs11	Juste avant sortie LLR	Plus direct, moins contexte
# Grad-CAM sur conv1	Début du réseau	Pour vérifier zones initialement activées

'''