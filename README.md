# Objectherkenning met de Computer Vision library Tensorflow

[Deze blogpost verscheen op Qdraw.nl](https://qdraw.nl/blog/design/objectherkenning-met-de-computer-vision-library-tensorflow/)

Met computer vision wordt het mogelijk om foto's en video's op intelligente wijze te analyseren om onze steden slimmer en veiliger te maken, nieuwe soorten robots te ondersteunen die productie optimaliseren. In dit artikel ligt een onderdeel van Computer Vision uit, namelijk Object Detection

## Tensorflow object detection api

Een belangrijke functionaliteit van Tensorflow is het '_image recognition'._ Tensorflow is een open source library dat door Google in 2015 voor het grote publiek beschikbaar is gemaakt. Het wordt gebruikt om deep learningmodels te bouwen, ontwerpen en te trainen.

Met Tensorflow is het mogelijk met de _object detection API_  wat het toegankelijk maakt voor onderzoekers en softwareontwikkelaars om objecten te identificeren in een 2d beeld. Het doel van Google van de _object detection API_  is om een evenwicht te hebben tussen simplicity en performance. Er zijn een aantal voorgetrainde modellen welke door wetenschappers worden gebruikt om algoritmes te trainen.

## En nu gaan we het gewoon zelf gebruiken!

Tensorflow is een library die beschikbaar is vanuit Python. Al het zware werk wordt buiten Python gedaan. Python is een programmeertaal die veel wordt gebruikt voor Machine Learning en data-analyse.

In deze tutorial ga ik ervan uit dat je een Ubuntu (Virtuele) machine tot je beschikking hebt. Buiten scope is het inrichten van Nvidia grafische kaarten en met CUDA. Ik heb deze code werkend op Ubuntu 16.04 en Mac OS Siera.

_De tekst gaat verder na de onderstaande afbeelding_

[![Objectherkenning met de Computer Vision library Tensorflow Uitzicht met Object Detection; object detection, uitzicht](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/500/20170725_160045_tensorflow-image_kl.jpg "Objectherkenning met de Computer Vision library Tensorflow Uitzicht met Object Detection | foto 1")](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/1000/20170725_160045_tensorflow-image_kl1k.jpg "Objectherkenning met de Computer Vision library Tensorflow Uitzicht met Object Detection | foto 1")

## PIP

Een Python package manager is PIP. Voor het gemak maak ik gebruik van Python 2.7, de code werkt ook met Python 3.6\. Voor deze tutorial gebruiken we deze en met het onderstaande commando installeer je deze op je systeem: Het dollarteken geeft aan dat een bash-commando is die als normale gebruiker moet worden uitgevoerd. Dit dollarteken hoeft niet mee te worden gekopieerd.

```shell
Ubuntu - $ sudo apt-get install python-pip git -y
```

```shell
Mac OS - $ brew install python
```

Ik clone de github responsories naar mijn home folder.

```shell
$  cd
```

## Tensorflow models

Clone the Tensorflow models van de officiële repository. In deze repository staan Machine Leaning-modelen die getraind zijn met Tensorflow. Voor deze tutorial gebruiken we alleen de _slim_ module en _object_detection_.

Met dit commando kopieer je de inhoud van de repository. De model map is 167,4 MB groot.

```shell
$  git clone https://github.com/tensorflow/models.git
```

Het het _pwd_ commando wordt het absolute path van de huidige map getoond.

```shell
$  pwd
```

```shell
/home/dion
```


Voeg onderaan toe waar _'/home/dion'_ de naam van je gebruiker is. Dit is om alle bestanden die in de map staan in python in het _path_ laden.

```shell
$  nano .profile
```

```shell
export PYTHONPATH=$PYTHONPATH:/home/dion/models:/home/dion/models/slim
```


Laad de inhoud van _.profile_ of herstart alle terminal vensters om de inhoud van het bestand te laden.

```shell
$  source .profile
```

## Protobuf

Google heeft een manier ontwikkeld om frozen models op te slaan. Een frozen model is Neural Network dat is opgeslagen en in het geheugen kan worden ingeladen.

Dit protocol moet eerst worden geïnstalleerd.

```shell
Ubuntu - $ sudo apt-get install protobuf-compiler -y
```

```shell
Mac OS - $ brew install protobuf
```

De Protobuf libraries moeten eerst worden gecompiled. Dit moet je doen vanuit de _model-map_:

```shell
$  cd models
```

```shell
$  protoc object_detection/protos/*.proto --python_out=.
```

En ik ga terug naar mijn home folder:
```shell
$  cd
```

## OpenCV

Met het onderstaande commando laad ik een bash script van een externe website waarbij OpenCV automatisch wordt gecompileerd en geïnstalleerd in _/usr/local._ Dit script installeert OpenCV 3.2 en werkt met Ubuntu 16.04.

```shell
$  curl -L [https://raw.githubusercontent.com/qdraw/tensorflow-object-detection-tutorial/master/install.opencv.ubuntu.sh](https://raw.githubusercontent.com/qdraw/tensorflow-object-detection-tutorial/master/install.opencv.ubuntu.sh) | bash
```

Op mijn Macbook maak ik gebruik van OpenCV 2.4\. Ik heb hier OpenCV 3.2 proberen te installeren allen dit werkt goed in combinatie met Mac OS Siera en Python 3.6

```shell
$  brew install homebrew/science/opencv
```

_De tekst gaat verder na de onderstaande afbeelding_

[![Objectherkenning met de Computer Vision library Tensorflow Compiling OpenCV @Ubuntu 16.04; OpenCV, Ubuntu](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/500/20170725_205133_compiling-opencv_kl.jpg "Objectherkenning met de Computer Vision library Tensorflow Compiling OpenCV @Ubuntu 16.04 | foto 2")](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/1000/20170725_205133_compiling-opencv_kl1k.jpg "Objectherkenning met de Computer Vision library Tensorflow Compiling OpenCV @Ubuntu 16.04 | foto 2")

## Tensorflow-object-detection-tutorial repository

In deze repository heb ik alle inhoud verzameld. Er zijn twee voorbeelden, het eerste voorbeeld wordt een afbeelding geanalyseerd en het tweede voorbeeld laat een livebeeld zien van de webcam. Met de onderstaande opdracht kopieer je de map van Github.

```shell
$  git clone https://github.com/qdraw/tensorflow-object-detection-tutorial.git
```

Het volgende deel van de uitleg voeren we uit vanuit de onderstaande map:
```shell
$  cd tensorflow-object-detection-tutorial/
```

De benodigdheden van de demo moeten nog worden geïnstalleerd.

```shell
$  pip install -r requirements.txt
```

## Het analyseren van een afbeelding

Voor deze demonstratie analyseren we een foto die gemaakt is bij Colours op het kantoor in Den Bosch. We zoeken naar alle objecten in deze foto. Het algoritme kan een aantal auto's al vinden.

```shell
$ python image_object_detection.py
```

Druk _'ctrl + C' _ binnen het terminalvenster om het programma af te sluiten

_De tekst gaat verder na de onderstaande afbeelding_

[![Objectherkenning met de Computer Vision library Tensorflow Herkennen van een Appel en Banaan; appel, banaan, object detection](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/500/20170725_220836-detect-appel-banaan_kl.jpg "Objectherkenning met de Computer Vision library Tensorflow Herkennen van een Appel en Banaan | foto 3")](https://media.qdraw.nl/log//objectherkenning-met-de-computer-vision-library-tensorflow/1000/20170725_220836-detect-appel-banaan_kl1k.jpg "Objectherkenning met de Computer Vision library Tensorflow Herkennen van een Appel en Banaan | foto 3")

## Het analyseren van het beeld van je webcam

Met OpenCV wordt het mogelijk om webcambeelden in Python in te laden. In dit script draaien meerdere processen tegelijk waardoor het afsluiten lastig is.

```shell
$  python webcam_object_detection.py
```

De snelste en makkelijkste manier om af te sluiten is in een ander terminalvenster het proces te killen.

```shell
Ubuntu - $ pkill python
```

```shell
Mac OS - $ pkill Python
```

![Objectherkenning met de Computer Vision library Tensorflow - Demo, uitzicht](https://raw.githubusercontent.com/qdraw/tensorflow-object-detection-tutorial/master/test_images/example_webcam_640px.gif "Objectherkenning met de Computer Vision library Tensorflow - Demo")

_Het analyseren van het beeld van je webcam._

Mocht de wereld van Computer Vision je interesse hebben gewekt, maar weet je nog niet hoe je dit kunt toepassen en heb je de nodige vragen? Stuur mij dan een [mailtje](http://qdraw.nl/contact.html) dan kunnen we een kopje koffie drinken.

[Deze blogpost verscheen op Qdraw.nl](https://qdraw.nl/blog/design/objectherkenning-met-de-computer-vision-library-tensorflow/)
