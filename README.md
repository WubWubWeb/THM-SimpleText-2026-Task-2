# Setup
Built with Python 3.15.5.

1. Create a file named `openai_token.txt` in the root directory and paste your ChatGPT token into it.
2. Create a file named `google_token.txt` in the root directory and paste your Gemini token into it.

3. Download the SimpleText 2026 task 2 dev-, test- and training data.
4. Create a folder named `Task_2_new` in the root directory and copy the downloaded data into it.

5. Run `python -m venv .venv` in the root directory. Using a virtual environment is always recommended.

# How to run
The notebook files contain cells with pip instructions. Running these cells downloads the required packages. If version conflicts arise at a later date, run `pip install requirements.txt` in the root directory using a terminal.

runner_colab.py is a file intended for use with google colab. Please note that this has only been tested up to the cell running the BERT models.

runner_local is intended for local use and depends on pytorch 2.6.0. Your setup may require using a different version. Edit the first cell containing the pip instructions if needed.

If you are planning on training the BERT models locally, adjust the output paths in `berthold_classifying.py`.