
# An-AI-Chatbot-in-Python-and-Flask
An AI Chatbot using Python and Flask REST API

## Requirements (libraries)
1. TensorFlow
2. Flask

## VsCode SetUp
1. Clone the repository -> cd into the cloned repository folder
2. Create a python virtual environment 
```
# macOS/Linux
# You may need to run sudo apt-get install python3-venv first
python3 -m venv .venv

# Windows
# You can also use py -3 -m venv .venv
python -m venv .venv
```
When you create a new virtual environment, a prompt will be displayed to allow you to select it for the workspace.

3. Activate the virtual environment
```
# Linux
source ./venv/bin/activate  # sh, bash, or zsh

# Windows
.
env\Scripts ctivate
```

4. Run `pip install --upgrade tensorflow` to install `TensorFlow`
5. Run `pip install -U nltk` to install `nltk`
6. Run `pip install -U Flask` to install `Flask`
7. To expose your bot via Ngrok, run `pip install flask-ngrok` to install `flask-ngrok`. Then you'll need to configure your ngrok credentials (login: email + password). Then uncomment this line `run_with_ngrok(app)` and comment the last two lines `if __name__ == "__main__": app.run()`. Notice that ngrok is not used by default.
8. To access your bot on localhost, go to `http://127.0.0.1:5000/`. If you're on Ngrok, your URL will be `some-text.ngrok.io`.

### Step-By-Step Explanation and Installation Guide
> https://dentricedev.com/blog/how-to-create-an-ai-chatbot-in-python-and-flask-gvub
> 
> https://dev.to/dennismaina/how-to-create-an-ai-chatbot-in-python-and-flask-1c3m

### Execution
To run this Bot, first, run the `train.py` file to train the model. This will generate a file named `chatbot_model.h5`.
This is the model that will be used by the Flask REST API to easily give feedback without the need to retrain.
After running `train.py`, next run the `app.py` to initialize and start the bot.
To add more terms and vocabulary to the bot, modify the `intents.json` file and add your personalized words and retrain the model again.

### Run the Project
1. Open terminal in the project directory.
2. Run `python train.py` to train and compile the model. This will generate the `chatbot_model.h5` file.
3. Once the model is trained, run `python app.py` to start the Flask application.
4. The app will be accessible at `http://127.0.0.1:5000/` locally. You can now interact with the chatbot via the API.
