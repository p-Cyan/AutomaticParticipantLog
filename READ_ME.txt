
NOTE: For the sake of participants' privacy, the data of old participants have been deleted. An folder of just me
exists as an example for adding future data.

NOTE: This project file was initially used as a DEMO before loading into main PC and connecting to the IP Camera stream so the file
lacks all of the extra configuration code.


INSTRUCTIONS
-----------------

1. Save participant images under data/train/participant_name filepath. ( "unknown" directory in data/ train is for
classifying unknown images, add all non participants or randoms to this group for better classification)
2. Run training_data_embeddings.py to generate embeddings for each participant. These are stored in save_files
3. Run main_file.py to start the program. Every 10 sec the program updates the LogBook.csv with logs of each participant.