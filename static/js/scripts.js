let selectedGameId = null;

async function searchGames() {
  const query = document.getElementById("query").value;
  const resultsDiv = document.getElementById("results"); // Clear previous results/plot
  resultsDiv.innerHTML = "";
  const gameList = document.getElementById("gameList");
  gameList.innerHTML = "<p>Searching...</p>"; // Indicate searching
  gameList.style.display = "block"; // Show the list area

  try {
    const response = await fetch("/search", {
      method: "POST",
      body: new URLSearchParams({ query: query }),
    });
    const data = await response.json();

    gameList.innerHTML = ""; // Clear searching message

    if (!data.games || data.games.length === 0) {
      gameList.innerHTML = "<p>No games found.</p>";
      document.getElementById("recommendSection").style.display = "none";
      // Keep gameList visible to show "No games found" message
      // gameList.style.display = "none"; // Optionally hide if no results found
      return;
    }

    // Display search results
    data.games.forEach((game) => {
      const div = document.createElement("div");
      div.className = "game-item";
      div.textContent = game.name;
      div.onclick = () => selectGame(game.appid, game.name);
      gameList.appendChild(div);
    });

    // Show recommendation section only after a successful search finds games
    document.getElementById("recommendSection").style.display = "block";
  } catch (error) {
    console.error("Search failed:", error);
    gameList.innerHTML = "<p>Search failed. Please try again.</p>";
    document.getElementById("recommendSection").style.display = "none";
  }
}

function selectGame(appid, name) {
  selectedGameId = appid;
  document.getElementById("selectedGame").value = name;
  document.getElementById("game_id").value = appid;
  const gameList = document.getElementById("gameList");
  gameList.innerHTML = ""; // Clear the list
  gameList.style.display = "none"; // Hide the list after selection
  document.getElementById("results").innerHTML = ""; // Clear previous results/plot
}

// --- Recommendation Form Submission ---
document.getElementById("recommendForm").addEventListener("submit", (e) => {
  e.preventDefault(); // 防止默认提交
  if (!selectedGameId) {
    alert("Please select a game first.");
    return;
  }

  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "<p>Loading recommendations and plot...</p>";
  resultsDiv.style.display = "block"; // 显示 results div

  const formData = new FormData(e.target);

  fetch("/recommend", {
    method: "POST",
    body: formData,
  })
    .then((response) => {
      if (!response.ok) {
        return response.json().then((errData) => {
          throw new Error(errData.error || response.statusText);
        });
      }
      return response.json();
    })
    .then((data) => {
      let recHtml = `<h2>Query Game: ${data.query_game}</h2><ul>`;
      data.recommendations.forEach((rec) => {
        const description = rec.Description
          ? rec.Description.slice(0, 200) +
            (rec.Description.length > 200 ? "..." : "")
          : "No description available.";
        recHtml += `<li><strong>${rec.Name}</strong> (ID: ${
          rec.ID
        }, Weighted Score: ${rec.Weighted_Score.toFixed(3)}, Review Number: ${
          rec.Num_of_reviews
        }, Similarity: ${rec.Similarity.toFixed(3)})<br>
                      Release: ${
                        rec.Release_Date || "N/A"
                      }, Rating: ${rec.Rating.toFixed(3)}, Genres: ${
          rec.Genres && rec.Genres.length > 0 ? rec.Genres.join(", ") : "N/A"
        }<br>
                    </li>`;
      });
      recHtml += `</ul>`;
      resultsDiv.innerHTML = recHtml;
    })
    .catch((error) => {
      resultsDiv.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
    });
});
