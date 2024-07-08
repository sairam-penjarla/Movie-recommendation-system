# Movie Recommendation System

This repository contains a movie recommendation system that uses a combination of data preprocessing, transformation, and a machine learning model to provide recommendations based on cosine similarity. The system is built using Python and the sklearn library.

## Features

- Data preprocessing to handle missing values and clean the data.
- Data transformation to create a 'soup' of features for each movie.
- Model training using CountVectorizer and cosine similarity.
- Recommendations based on the trained model.

## Installation

1. Clone the repository:

```bash
git clone https://github.com/yourusername/movierecommendationsystem.git
cd movierecommendationsystem
```

2. Create a virtual environment and activate it:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```
## Configuration

Make sure you have a `config.yaml` file in the root directory. Refer to the structure and parameters defined in the example provided in the repository to set up your configuration file.

## Usage

1. Ensure your configuration file `config.yaml` is set up correctly with paths and parameters.

2. Run the following script to load data, preprocess it, train the model, and get movie recommendations:

```bash
python main.py
```

3. Check the log file specified in the `config.yaml` for detailed logs of the processing and model training.

## Example

To get recommendations for a specific movie, you can modify the `main.py` to include the movie title you want recommendations for:

```python
print(recommender.get_recommendations('PK'))
```

Replace `'PK'` with any movie title of your choice. 

## Logging

The application logs all steps including data loading, preprocessing, and model training. Logs are saved to a file specified in the `config.yaml` and also printed to the console.

## Contributing

If you want to contribute to this project, please fork the repository and submit pull requests. Ensure your code follows the style and conventions used in this repository.

Happy coding!