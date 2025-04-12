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
      // 构建推荐列表 HTML
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
        "${rec.Image_url}"<br>
                    </li>`;
      });
      recHtml += `</ul>`;
      resultsDiv.innerHTML = recHtml;

      // 使用返回的 plot_data 数据绘制 3D 图表
      const plotData = data.plot_data;
      // 将数据分成 Query 和 Recommendation 两部分
      const queryPoints = plotData.filter((pt) => pt.Type === "Query");
      const recPoints = plotData.filter((pt) => pt.Type === "Recommendation");

      // 构造 Query trace
      const traceQuery = {
        x: queryPoints.map((pt) => pt.PC1),
        y: queryPoints.map((pt) => pt.PC2),
        z: queryPoints.map((pt) => pt.PC3),
        mode: "markers+text",
        type: "scatter3d",
        name: "Query",
        marker: { size: 12, color: "red" },
        text: queryPoints.map((pt) => pt.Name),
        textposition: "top center",
      };

      // 构造 Recommendation trace
      const traceRec = {
        x: recPoints.map((pt) => pt.PC1),
        y: recPoints.map((pt) => pt.PC2),
        z: recPoints.map((pt) => pt.PC3),
        mode: "markers+text",
        type: "scatter3d",
        name: "Recommendation",
        marker: { size: 8, color: "blue" },
        text: recPoints.map((pt) => pt.Name),
        textposition: "top center",
      };

      const dataPlot = [traceQuery, traceRec];

      const layout = {
        title: "Query & Recommendations (3D Visualization)",
        scene: {
          xaxis: { title: "PC1" },
          yaxis: { title: "PC2" },
          zaxis: { title: "PC3" },
        },
        width: 800,
        height: 600,
      };

      // 渲染图表到预留的 plot 容器中
      const plotDiv = document.getElementById("plot");
      Plotly.newPlot(plotDiv, dataPlot, layout);
    })
    .catch((error) => {
      resultsDiv.innerHTML = `<p style="color: red;"><strong>Error:</strong> ${error.message}</p>`;
    });
});
