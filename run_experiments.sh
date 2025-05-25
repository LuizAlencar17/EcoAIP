# #!/bin/bash

python main.py --config config.improved_aip_resnet50 --mode train
python main.py --config config.resnet50 --mode train
python main.py --config config.aip_resnet50 --mode train

python main.py --config config.improved_aip_resnet50 --mode test
python main.py --config config.resnet50 --mode test
python main.py --config config.aip_resnet50 --mode test