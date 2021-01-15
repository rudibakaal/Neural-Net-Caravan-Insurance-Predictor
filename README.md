# Neural-Net-Caravan-Insurance-Predictor

## Motivation
Neural network trained on data garnered from the CoIL 2000 Challenge to predict caravan insurance policy holders.

The data consists of 86 variables and includes product usage data and socio-demographic data derived from zip area codes. The data was collected to answer the following question: Can you predict who would be interested in buying a caravan insurance policy?[1].

## Neural Network Topology and Results Summary
The binary-crossentropy loss function was leveraged along with the rmsprop optimizer for this classification task.

![model](https://user-images.githubusercontent.com/48378196/96961401-4be81500-1550-11eb-9cd2-4e0f682c3b56.png)

By the 60th epoch, binary and validation classifiers reach 97% and 91% accuracy respectively in correctly identifying caravan insurance policy holders.

![caravan_insurance](https://user-images.githubusercontent.com/48378196/104682788-a22cb480-5749-11eb-9633-d7fdcd93e04c.png)

## License
[MIT](https://choosealicense.com/licenses/mit/) 

## References
[1] P. van der Putten and M. van Someren (eds). CoIL Challenge 2000: The Insurance Company Case. Published by Sentient Machine Research, Amsterdam. Also a Leiden Institute of Advanced Computer Science Technical Report 2000-09. June 22, 2000.
 
