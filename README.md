# Mini_Rogue_V1

 First implementation of my own AI built from scratch that will one day be Rogue.

 the files can be tweaked first, right at the moment they are set to run and produce a small model as a trial run to make sure I have everything built correct.  Once I know it is, I will share this and go on to advance my model.

 1. Put your dataset in the data folder and on line 51 of the data_prep.py file put in the path to the file.  if you get everything right, it will output two files.  a vocab.txt for your reference, and a cleaned_text.txt file.  BACK UP the files just incase... For some reason I keep having the contents vanish if I wait too long to run the training loop... so that's what the backup folder is for (there is a data folder and a model folder to backup the model parameters file you will get after all the training is done)

 2. Now, add the path of the cleaned_text.txt to the train_rogue.py file on line 93, and make any tweaks you would like to both the train_rogue.py and the mini_rogue_v1.py files.  Change learning rates, the model parameters, go wild.  Remember, it is a balance between all of them to get a decaying loss value, the time it will take to train, and the compute resources you have to throw at it.  ALSO, edit the files to add your compute info if need be...when you are one, you will have one mini_rogue_model.pth file that contains your newly trained model!
