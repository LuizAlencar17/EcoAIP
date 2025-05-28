# #!/bin/bash

# # serengeti
# python main.py --config config.serengeti.aip_resnet50 --mode train
# python main.py --config config.serengeti.improved_aip_resnet50 --mode train
# python main.py --config config.serengeti.resnet50 --mode train

# python main.py --config config.serengeti.aip_resnet50 --mode test
# python main.py --config config.serengeti.improved_aip_resnet50 --mode test
# python main.py --config config.serengeti.resnet50 --mode test

# caltech
python main.py --config config.caltech.aip_resnet50_1200 --mode train
python main.py --config config.caltech.improved_aip_resnet50_1200 --mode train
python main.py --config config.caltech.resnet50_1200 --mode train

python main.py --config config.caltech.aip_resnet50_1200 --mode test
python main.py --config config.caltech.improved_aip_resnet50_1200 --mode test
python main.py --config config.caltech.resnet50_1200 --mode test

python main.py --config config.caltech.aip_resnet50_2400 --mode train
python main.py --config config.caltech.improved_aip_resnet50_2400 --mode train
python main.py --config config.caltech.resnet50_2400 --mode train

python main.py --config config.caltech.aip_resnet50_2400 --mode test
python main.py --config config.caltech.improved_aip_resnet50_2400 --mode test
python main.py --config config.caltech.resnet50_2400 --mode test

python main.py --config config.caltech.aip_resnet50_4800 --mode train
python main.py --config config.caltech.improved_aip_resnet50_4800 --mode train
python main.py --config config.caltech.resnet50_4800 --mode train

python main.py --config config.caltech.aip_resnet50_4800 --mode test
python main.py --config config.caltech.improved_aip_resnet50_4800 --mode test
python main.py --config config.caltech.resnet50_4800 --mode test