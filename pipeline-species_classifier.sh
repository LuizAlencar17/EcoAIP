# !/bin/bash

# ---------------------------------------- SERENGETI

# serengeti 1200
python main.py --config config.species_classifier.serengeti.aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.serengeti.resnet50_1200 --mode train

python main.py --config config.species_classifier.serengeti.aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.serengeti.resnet50_1200 --mode test

# serengeti 2400
python main.py --config config.species_classifier.serengeti.aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.serengeti.resnet50_2400 --mode train

python main.py --config config.species_classifier.serengeti.aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.serengeti.resnet50_2400 --mode test

# serengeti 4800
python main.py --config config.species_classifier.serengeti.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.serengeti.resnet50_4800 --mode train

python main.py --config config.species_classifier.serengeti.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.serengeti.resnet50_4800 --mode test

# # serengeti 9600
# python main.py --config config.species_classifier.serengeti.aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.serengeti.resnet50_9600 --mode train

# python main.py --config config.species_classifier.serengeti.aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.serengeti.improved_aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.serengeti.resnet50_9600 --mode test


# ---------------------------------------- CALTECH

# caltech 1200
python main.py --config config.species_classifier.caltech.aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.caltech.resnet50_1200 --mode train

python main.py --config config.species_classifier.caltech.aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.caltech.resnet50_1200 --mode test

# caltech 2400
python main.py --config config.species_classifier.caltech.aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.caltech.resnet50_2400 --mode train

python main.py --config config.species_classifier.caltech.aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.caltech.resnet50_2400 --mode test

# caltech 4800
python main.py --config config.species_classifier.caltech.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.caltech.resnet50_4800 --mode train

python main.py --config config.species_classifier.caltech.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.caltech.improved_aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.caltech.resnet50_4800 --mode test

# # caltech 9600
# python main.py --config config.species_classifier.caltech.aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.caltech.improved_aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.caltech.resnet50_9600 --mode train

# python main.py --config config.species_classifier.caltech.aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.caltech.improved_aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.caltech.resnet50_9600 --mode test

# ---------------------------------------- WCS

# wcs 1200
python main.py --config config.species_classifier.wcs.aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_1200 --mode train
python main.py --config config.species_classifier.wcs.resnet50_1200 --mode train

python main.py --config config.species_classifier.wcs.aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_1200 --mode test
python main.py --config config.species_classifier.wcs.resnet50_1200 --mode test

# wcs 2400
python main.py --config config.species_classifier.wcs.aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_2400 --mode train
python main.py --config config.species_classifier.wcs.resnet50_2400 --mode train

python main.py --config config.species_classifier.wcs.aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_2400 --mode test
python main.py --config config.species_classifier.wcs.resnet50_2400 --mode test

# wcs 4800
python main.py --config config.species_classifier.wcs.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier.wcs.resnet50_4800 --mode train

python main.py --config config.species_classifier.wcs.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.wcs.improved_aip_resnet50_4800 --mode test
python main.py --config config.species_classifier.wcs.resnet50_4800 --mode test

# # wcs 9600
# python main.py --config config.species_classifier.wcs.aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.wcs.improved_aip_resnet50_9600 --mode train
# python main.py --config config.species_classifier.wcs.resnet50_9600 --mode train

# python main.py --config config.species_classifier.wcs.aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.wcs.improved_aip_resnet50_9600 --mode test
# python main.py --config config.species_classifier.wcs.resnet50_9600 --mode test