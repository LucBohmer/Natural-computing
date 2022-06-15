To start the training, run tournament.py, which will start the process and start dumping sets of weights into weights.json
It will also output the average and best score of the generation after each generation.
If you want to continue on a previuosly trained generation, set start_from_file to True in the main function. 
If you want to change the number of used phases to 2 instead of 3, this can be set in the phases parameter in the main function.
