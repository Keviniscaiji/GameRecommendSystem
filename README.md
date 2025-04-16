# 🎮 Steam Game Recommendation System

A Flask-based application that generates Steam user profiles and provides game recommendations using precomputed game embeddings.

---

## 🚀 Setup Instructions

### 1. **Clone the Repository**

```bash
git clone https://github.com/Keviniscaiji/GameRecommendSystem.git
cd steam-game-recommender
```

---

### 2. **Set Up a Virtual Environment (Recommended)**

To isolate project dependencies and avoid conflicts, it's highly recommended to use a Python virtual environment:

```bash
python -m venv .venv
```

Then activate the environment:

- **macOS/Linux:**

  ```bash
  source .venv/bin/activate
  ```

- **Windows:**

  ```cmd
  .venv\Scripts\activate
  ```

Once activated, install dependencies:

```bash
pip install -r requirements.txt
```

> ✅ Tip: Add `.venv/` to your `.gitignore` to avoid committing the virtual environment to version control.

---

### 3. **Download the Embedding File**

Download the precomputed game embeddings from the following link:  
[📁 game_embeddings.pkl (Google Drive)](https://drive.google.com/file/d/1xPRRwfSXDQE5OsH5dwK41xwcWrLt2VOE/view?usp=sharing)

Place the downloaded file in the **root directory** of the project.

---

### 4. **Set Up Environment Variables**

This project uses a `.env` file to securely manage your Steam API key.

#### Option 1: Manual Setup

1. Create a `.env` file in the root directory.
2. Add the following line:

   ```env
   STEAM_API_KEY=your-steam-api-key-here
   ```

   You can obtain a key from the [Steam Web API](https://steamcommunity.com/dev/apikey).

#### Option 2: Quick Setup

```bash
echo "STEAM_API_KEY=your-steam-api-key-here" > .env
```

> 🔒 **Important:**  
> - Never commit your `.env` file to version control.  
> - Ensure `.env` is listed in `.gitignore`.

---

### 5. **Run the Backend Server**

Once everything is set up, start the Flask server:

```bash
python app.py
```

By default, the app runs at: [http://localhost:5001](http://localhost:5001)

---

### 6. **Access the Application**

Open your browser and go to:

```
http://localhost:5001
```

Enter a Steam ID to generate the user profile and view personalized game recommendations powered by the embedding model.
