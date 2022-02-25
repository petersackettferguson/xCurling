# xCurling
xG for curling.

Dependencies: scikit-learn, opencv

xCurling uses data about curling stone positions and attempted shots to build a model in order to attempt to predict the average accuracy of a given attempt. The data is processed using sklearn, with multiple possible models (SVM seems to produce the best results with limited data). OpenCV is used to analyze still overhead frames of curling sheets to facilitate data entry.
