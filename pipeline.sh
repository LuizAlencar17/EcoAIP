
# # ---------------------------------------- CLASSIFICATION ----------------------------------------

# # ---------------------------------------- SERENGETI
python main.py --config config.species_classifier_cropped.serengeti.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.serengeti.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.serengeti.eco_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.serengeti.eco_aip_resnet50_4800 --mode test

# # ---------------------------------------- WCS
python main.py --config config.species_classifier_cropped.wcs.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.wcs.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.wcs.eco_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.wcs.eco_aip_resnet50_4800 --mode test

# # ---------------------------------------- CALTECH
python main.py --config config.species_classifier_cropped.caltech.resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.caltech.aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.aip_resnet50_4800 --mode test
python main.py --config config.species_classifier_cropped.caltech.eco_aip_resnet50_4800 --mode train
python main.py --config config.species_classifier_cropped.caltech.eco_aip_resnet50_4800 --mode test

# # # ---------------------------------------- NACTI
# python main.py --config config.species_classifier_cropped.nacti.resnet50_1200 --mode train
# python main.py --config config.species_classifier_cropped.nacti.resnet50_1200 --mode test
# python main.py --config config.species_classifier_cropped.nacti.aip_resnet50_1200 --mode train
# python main.py --config config.species_classifier_cropped.nacti.aip_resnet50_1200 --mode test
# python main.py --config config.species_classifier_cropped.nacti.eco_aip_resnet50_1200 --mode train
# python main.py --config config.species_classifier_cropped.nacti.eco_aip_resnet50_1200 --mode test

# ---------------------------------------- DETECTATION ----------------------------------------

# # # ---------------------------------------- SERENGETI
# python main.py --config config.detectation_classifier.serengeti.yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.serengeti.yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.serengeti.aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.serengeti.aip_yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.serengeti.eco_aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.serengeti.eco_aip_yolov11_4800 --mode test

# # ---------------------------------------- WCS
# python main.py --config config.detectation_classifier.wcs.yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.wcs.yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.wcs.aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.wcs.aip_yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.wcs.eco_aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.wcs.eco_aip_yolov11_4800 --mode test

# # ---------------------------------------- CALTECH
# python main.py --config config.detectation_classifier.caltech.yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.caltech.yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.caltech.aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.caltech.aip_yolov11_4800 --mode test
# python main.py --config config.detectation_classifier.caltech.eco_aip_yolov11_4800 --mode train
# python main.py --config config.detectation_classifier.caltech.eco_aip_yolov11_4800 --mode test

# # ---------------------------------------- NACTI
# python main.py --config config.detectation_classifier.nacti.yolov11_1200 --mode train
# python main.py --config config.detectation_classifier.nacti.yolov11_1200 --mode test
# python main.py --config config.detectation_classifier.nacti.eco_aip_yolov11_1200 --mode train
# python main.py --config config.detectation_classifier.nacti.eco_aip_yolov11_1200 --mode test
# python main.py --config config.detectation_classifier.nacti.aip_yolov11_1200 --mode train
# python main.py --config config.detectation_classifier.nacti.aip_yolov11_1200 --mode test

