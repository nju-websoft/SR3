from getData import *
addRetrieveData('../data/train_raw.json', '../data/train_retrieve.json')
addRetrieveData('../data/val_raw.json', '../data/val_retrieve.json')
addRetrieveData('../data/test_raw.json', '../data/test_retrieve.json')
tagGoldInfo('../data/test_retrieve.json','../data/test.json')
tagGoldInfo('../data/val_retrieve.json','../data/val.json')
tagGoldInfo('../data/train_retrieve.json','../data/train.json')
generateClassifyERNIEBysent('../data/train.json','../data/train.tsv')
generateClassifyERNIEBysent('../data/val.json','../data/val.tsv')
generateClassifyERNIEBysent('../data/test.json','../data/test.tsv')
