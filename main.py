import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("Avant import head_mimo")
from head_mimo import *
print("Après import head_mimo")

if __name__ == '__main__':
    print("Entrée dans main")
    args = get_config()
    print("Après get_config")
    log(args)
    print("Après log(args)")

    # model
    model = mmsenet(args)
    print("Après création modèle")

    # main
    if args.phase == 'train':
        print("Phase train")
        train = Trainer(args, model)
        train.tr()
    elif args.phase == 'test':
        print("Phase test")
        test = Tester(args, model)
        test.test()

    print('[*] Finish!')
 #tester avec zero en shape pour voir le probleme si je peut entrainner 