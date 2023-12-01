# BERT Code for Podcast project
Step 1: Change path to csv file in params.yaml

Step 2: Run survey code to choose best epoch numbers for first 512 tokens and last 512 tokens:
```
mkdir logs
./survey.sh
```
The logs are saved in *logs/filename.json*, you need to process the json files to get the best number

Step 3: Run train code to get the model
```
./train.sh
```
