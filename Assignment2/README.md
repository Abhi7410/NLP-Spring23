<h1>README</h1>

This is a Python script for performing part-of-speech tagging on input sentences. It uses a Bidirectional LSTM model for tagging words in the input sentence with their corresponding parts of speech.

1. Download the pre-trained model file from following link:

   [modelFinal.pt](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/abhishek_shar_students_iiit_ac_in/EQolQYVa_d9HikFfdb9hSAkBmJYWazcov67COJ5CY9CBiw?e=NJyRi8)

2. Download the embedding matrix (pre-trained) 

   [embeddings_matrix.npy](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/abhishek_shar_students_iiit_ac_in/EQTiSLt110BHiOuHRCul8CoB3RKBw2YTqK09lIi2gfyJsg?e=Kfg3hs)

2. Place the downloaded files in the parent directory.
3. Ensure that you have all required dependencies installed .
   * Numpy
   * Torch
   * Conllu

4. Run the python program :

   ```bash
   python3 pos_tagger.py
   ```

5. Once the script is running , wait for few seconds to load the embedding matrix (Glove) , enter a sentence without using punctuation marks when prompted.
6. The script will then tag each word in the sentence with its corresponding part of speech, and print out the results in the format of `word <tab> tag`.