# Steam Profile Generator and Game Recommendation System

A Flask-based application that generates Steam user profiles and provides game recommendations using precomputed game embeddings.

## 🚀 Setup Instructions

### 1. **Download the Embedding File**
Download the embedding file from the following link:  
[game_embeddings.pkl (Google Drive)](https://drive.google.com/file/d/1xPRRwfSXDQE5OsH5dwK41xwcWrLt2VOE/view?usp=sharing)  
After downloading, place the file in the root directory of your project.

### 2. **Set Up Environment Variables**
This project uses a `.env` file to securely manage sensitive information like the Steam API key. Follow these steps:
- Create a file named `.env` in the root directory of your project.
- Add your Steam API key to the `.env` file in the following format:
  ```
  STEAM_API_KEY=your-steam-api-key-here
  ```
  Replace `your-steam-api-key-here` with your actual Steam Web API key (get it from [Steam Web API](https://steamcommunity.com/dev/apikey)).
- Ensure you have the `python-dotenv` package installed (included in `requirements.txt`).

**Note**: Do not commit the `.env` file to version control. Add it to `.gitignore` to keep your API key private.

### 3. **Install Dependencies**
Ensure you have Python 3 installed. Then run:

```bash
pip install -r requirements.txt
```

This will install all required packages, including `flask`, `requests`, `python-dotenv`, and others.

### 4. **Run the Backend Server**
Start the Flask application:

```bash
python app.py
```

The server will start on `http://localhost:5001` by default.

### 5. **Access the Application**
Open your browser and navigate to:  
[http://localhost:5001](http://localhost:5001)

You can input a Steam ID to generate a user profile and view game recommendations based on the precomputed embeddings.
