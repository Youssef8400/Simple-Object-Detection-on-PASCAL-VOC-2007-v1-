### Détection d’Objets Simplifiée - VOC2007

##  Installation :
```bash
pip install -r requirements.txt
```

## Exécution de application :
```python
python deploiement.py
```


---


### Description

Ce projet met en place un modèle de détection d’objets entraîné sur le dataset PASCAL VOC 2007 à l’aide de TensorFlow et TensorFlow Datasets.
Il utilise un réseau de neurones convolutionnel (CNN) avec deux sorties :

bbox : coordonnées normalisées de la bounding box

Une interface Tkinter permet de charger une image depuis l’ordinateur et d’afficher le résultat de la prédiction avec la bounding box dessinée .

Ce projet est conçu comme une version simplifiée d’un système de détection d’objets, idéale pour comprendre :

Les bases du traitement d’images

L’entraînement de modèles CNN

L’intégration d’un modèle dans une interface graphique Python




### Test 


# Test 1 :

<img width="750" height="792" alt="image" src="https://github.com/user-attachments/assets/0bb16233-ee1c-4455-8439-b5627be6a182" />



# Test 2 :

<img width="373" height="391" alt="chat2" src="https://github.com/user-attachments/assets/0cd55483-2a00-43b6-9632-9e610baf65f2" />



###  Limites et améliorations

| Limites actuelles                                                                 | Pistes d’amélioration possibles                                                                 |
|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|
| Modèle simple (version 1) basé sur un petit CNN                                   | Utiliser un modèle pré-entraîné type **YOLO**, **SSD** ou **Faster R-CNN** pour de meilleures performances |
| Fonctionne sur une seule image à la fois                                          | Ajouter un traitement par lot ou un mode webcam en temps réel                                   |
| Précision moyenne, sensible aux images complexes                                  | Augmenter la profondeur du réseau et le temps d’entraînement                                    |
| Ne gère pas la détection multi-objets                                             | Adapter la sortie pour plusieurs bounding boxes par image                                       |
| Dataset limité (VOC 2007 uniquement)                                              | Enrichir avec **VOC 2012**, **COCO** ou un dataset personnalisé                                 |


