
# # ---------------------------------------- DETECTATION ----------------------------------------

# # ---------------------------------------- NACTI
# python main.py --config config.detectation_classifier.nacti.yolov11_1200 --mode train
# python main.py --config config.detectation_classifier.nacti.yolov11_1200 --mode test

# # ---------------------------------------- WCS
# python main.py --config config.detectation_classifier.wcs.yolov11_10000 --mode train
# python main.py --config config.detectation_classifier.wcs.yolov11_10000 --mode test

# # ---------------------------------------- SERENGETI
# python main.py --config config.detectation_classifier.serengeti.yolov11_10000 --mode train
# python main.py --config config.detectation_classifier.serengeti.yolov11_10000 --mode test



# # ---------------------------------------- CLASSIFICATION ----------------------------------------

# # ---------------------------------------- SERENGETI
python main.py --config config.species_classifier_cropped.serengeti.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.serengeti.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.serengeti.improved_aip_resnet50_4800 --mode test

# # ---------------------------------------- CALTECH
python main.py --config config.species_classifier_cropped.caltech.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.caltech.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.caltech.improved_aip_resnet50_4800 --mode test

# # ---------------------------------------- WCS
python main.py --config config.species_classifier_cropped.wcs.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.improved_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.wcs.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.wcs.improved_aip_resnet50_4800 --mode test

# # ---------------------------------------- NACTI
python main.py --config config.species_classifier_cropped.nacti.resnet50_1200 --mode train
python main.py --config config.species_classifier_cropped.nacti.aip_resnet50_1200 --mode train
python main.py --config config.species_classifier_cropped.nacti.improved_aip_resnet50_1200 --mode train
python main.py --config config.species_classifier_cropped.nacti.resnet50_1200 --mode test
python main.py --config config.species_classifier_cropped.nacti.aip_resnet50_1200 --mode test
python main.py --config config.species_classifier_cropped.nacti.improved_aip_resnet50_1200 --mode test