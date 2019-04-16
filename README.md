# chat-bot
simple question and answer chat bot using keras, tensorflow backend


An end-to-end neural network was used, as outlined in the following paper: https://arxiv.org/pdf/1503.08895.pdf
The Babi data set from facebook research was used for training https://research.fb.com/downloads/babi/

The program takes in questions built from its vocab list, and returns (if all works) either yes or no. 

Due to known issues with saving and loading keras model weights between sessions, currently the bot interface is coded in the same file as 
the one that trains it. Once I have found a fix for loading model weights, I will upload a number of different trained model weights and allow
the user to select which one that their qna bot will use without them having to sit through model training. 
