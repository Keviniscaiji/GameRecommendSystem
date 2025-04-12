# Steam Game Recommendation System

A Flask-based application that generates Steam user profiles and provides game recommendations using precomputed game embeddings.

## 🚀 Setup Instructions

### 1. **Download the Embedding File**
Download the embedding file from the following link:  
[game_embeddings.pkl (Google Drive)](https://drive.google.com/file/d/1xPRRwfSXDQE5OsH5dwK41xwcWrLt2VOE/view?usp=sharing)  
After downloading, place the file in the root directory of your project.

### 2. **Set Up Environment Variables**

This project uses a `.env` file to securely manage sensitive information like the Steam API key. Follow the steps below:

#### 📄 Manual Setup

1. In the root directory of your project, create a `.env` file.
2. Add your Steam API key in the following format:

   ```env
   STEAM_API_KEY=your-steam-api-key-here
   ```

   Replace `your-steam-api-key-here` with your actual Steam Web API key. You can obtain one from the [Steam Web API](https://steamcommunity.com/dev/apikey).

#### ⚡ Quick Setup (Recommended)

Run this command in your terminal:

```bash
echo "STEAM_API_KEY=your-steam-api-key-here" > .env
```

> Make sure to replace the placeholder with your actual API key.

#### 🚫 Important

- **Do not** commit the `.env` file to version control.
- Ensure `.env` is listed in `.gitignore`.

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
