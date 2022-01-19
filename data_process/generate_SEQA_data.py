from getData import *
reloadERNIEscore('../data/val.json',
                 '../data/val.tsv',
                 '../data/val_result.0.0',
                 '../data/val_score.json')
reloadERNIEscore('../data/test.json',
                 '../data/test.tsv',
                 '../data/test_result.0.0',
                 '../data/test_score.json')
reloadERNIEscore('../data/train.json',
                 '../data/train.tsv',
                 '../data/train_result.0.0',
                 '../data/train_score.json')

generateGraphfile('../data/val_score.json',
                  '../data/val_graph.json')
generateGraphfile('../data/test_score.json',
                  '../data/test_graph.json')
generateGraphfile('../data/train_score.json',
                  '../data/train_graph.json')